//! Generic σ·p G-tensor assembler — device `#[cube]` (Phase 28, FND-05 / Gap B2).
//!
//! Emits the four `gc_x / gc_y / gc_z / gc_1` cartesian blocks that the host
//! spin-included transform `cart_to_spinor_si_2d` reads in order. Faithful port
//! of libcint `CINTgout1e_int1e_sp` (`autocode/intor3.c:416-440`) composed with
//! the overlap base G-tensor (`g1e.c`) and the bra-direction nabla
//! `CINTnabla1i_1e` (`g1e.c:322,345`).
//!
//! # Algorithm (per contracted shell pair i, j)
//! 1. Build the OVERLAP base G-tensor `g0` (fixed-center VRR + HRR), exactly as
//!    the gradient kernel (`one_electron.rs::one_electron_grad_bra_kernel`).
//! 2. Apply the bra nabla `g1 = nabla_i(g0)`:
//!    ```text
//!    ai2 = -2*ai;
//!    g1[off]      (ix==0) = ai2 * g0[off+1];
//!    g1[off+ix]   (ix>=1) = ix * g0[off+ix-1] + ai2 * g0[off+ix+1];
//!    ```
//! 3. Form the 3 Pauli components per cart `n`:
//!    ```text
//!    s[0] = g1x * g0y * g0z;   // -> gc_x
//!    s[1] = g0x * g1y * g0z;   // -> gc_y
//!    s[2] = g0x * g0y * g1z;   // -> gc_z
//!    ```
//!    For `int1e_sp` (`tensor_rank == 1`) the scalar slot `gc_1 == 0.0`.
//!
//! # Output layout (component-LEADING / pre-blocked — Spike Target C)
//! The four gc blocks are emitted **pre-blocked** (component-leading), NOT
//! component-interleaved. For contraction block `(ci, cj)` the base offset is
//! `(ci*nctr_j + cj) * (n_gc * block_len)` where `n_gc = 4` and
//! `block_len = nci*ncj`; within that block component `comp` occupies
//! `comp * block_len + (cj_idx * nci + ci_idx)` (KET-major `elem`). This is the
//! `gc[comp*(nf*ictr*jctr) + n]` layout libcint reaches via `CINTdmat_transpose`
//! (`cint1e.c:157`) — emitting it on-device avoids the host transpose, so
//! `cart_to_spinor_si_2d` reads `gc_x = block0, gc_y = block1, gc_z = block2,
//! gc_1 = block3` in order. (The separate KET→BRA orientation transpose is owned
//! inside the host transform.)
//!
//! # D-03 reusability
//! The assembler is parameterized by `#[comptime] tensor_rank`. For `int1e_sp`
//! `tensor_rank == 1` (3 Pauli + 1 zero scalar → `n_gc = 4` blocks). Phase 29's
//! `int1e_sigma` (`tensor_rank == 3`, 12-component) reuses the same gc-block
//! packing: it emits `tensor_rank * 4` blocks through the identical per-cart Pauli
//! mix, so the host transform / si_2d path is shared across the whole σ-group.

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
use crate::kernels::one_electron::{
    ONE_E_DERIV_SHAPE_STRIDE, OneEDerivLaunchGroup, one_e_deriv_single_pair_group,
    one_e_g_slab_stride, one_e_launch_geometry, one_e_per_unit,
};
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::transform::c2s::ncart;
use crate::transform::c2spinor::{cart_to_spinor_si_2d, cart_to_spinor_si_2di, spinor_len};
use cintx_core::CintFloat;
use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// sqrt(pi) — G-tensor base-case normalization (matches `g1e.c` `SQRTPI`).
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Number of gc blocks emitted per tensor component for the σ·p families:
/// `gc_x, gc_y, gc_z, gc_1` (3 Pauli + 1 scalar). For `int1e_sp` the scalar
/// block is identically zero (`CINTgout1e_int1e_sp`, `intor3.c:431-434`).
const N_GC: u32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
//  Shared per-axis VRR / HRR helpers (overlap base G-tensor).
//
//  Identical recurrences to `one_electron.rs::one_electron_vrr_axis` /
//  `one_electron_hrr_axis`; duplicated here to keep `sigma_p` self-contained
//  (the gradient kernel's copies are private to that module).
// ─────────────────────────────────────────────────────────────────────────────

/// Per-axis overlap VRR into the `g` sub-block starting at `base`.
#[cube]
fn sigma_p_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, rijrx: F, aij2: F, nmax: u32) {
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

/// Per-axis HRR into the `g` sub-block starting at `base` (i-stride = 1).
#[cube]
fn sigma_p_hrr_axis<F: Float>(g: &mut Array<F>, base: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
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

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch, launch_unchecked)]` — generic σ·p G-tensor assembler.
//
//  Single UNIT_POS==0 work item; in-kernel pdata recompute in F (norm_i =
//  norm_j = 1.0); zeroed scratch + out; (pi,pj) primitive loop × (ci,cj)
//  contraction loop — cloning `one_electron_grad_bra_kernel`'s structure.
//
//  G-tensor sizing: nmax = li+lj+1, lj_ext = lj (one extra bra level for the
//  ix+1 nabla read). The output is `N_GC`-block COMPONENT-LEADING per (ci,cj)
//  block — see module docs.
// ─────────────────────────────────────────────────────────────────────────────

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_p_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    gc_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] tensor_rank: u32,
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
        // Both scratch slabs share a stride and a slot, because the contraction
        // reads them at identical relative offsets.
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

            // G-tensor sizing (overlap base, one extra bra level for the nabla read).
            let nmax = li + lj + 1u32;
            let lj_ext = lj;
            let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            // n_gc = N_GC gc blocks per tensor component.
            let n_blocks = tensor_rank * N_GC;
            let total_len = n_blocks * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // Zero the full output buffer (the scalar gc_1 block is written 0 too).
            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
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

                    sigma_p_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    if lj_ext >= 1u32 {
                        sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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

                                            // 3 Pauli components (G1E_D_I mix).
                                            let s0 = g1x * g0y * g0z; // → gc_x
                                            let s1 = g0x * g1y * g0z; // → gc_y
                                            let s2 = g0x * g0y * g1z; // → gc_z

                                            let elem = cj_idx * nci + ci_idx;

                                            // Pre-blocked, component-LEADING write. For
                                            // int1e_sp (tensor_rank==1) the four gc
                                            // blocks are x,y,z then a ZERO scalar slot.
                                            // For tensor_rank>1 the per-tensor scalar
                                            // term is supplied by future σ families
                                            // (e.g. int1e_sigma) — int1e_sp keeps it 0.
                                            let bx = base + elem;
                                            let by = base + block_len + elem;
                                            let bz = base + 2u32 * block_len + elem;
                                            let b1 = base + 3u32 * block_len + elem;
                                            gc_out[bx as usize] += weight * s0;
                                            gc_out[by as usize] += weight * s1;
                                            gc_out[bz as usize] += weight * s2;
                                            // gc_1 (scalar slot) stays 0.0 for int1e_sp.
                                            gc_out[b1 as usize] += F::new(0.0_f32);

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

/// `g_per_axis` for one σ·p class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn sigma_p_g_per_axis(li: usize, lj: usize) -> usize {
    // Overlap base with one extra bra level: nmax = li+lj+1, lj_ext = lj.
    (li + lj + 2) * (lj + 1)
}

/// Evaluate every launch group of a batched σ·p run (Task 35-D, wave 5).
///
/// `tensor_rank` is fixed by the caller's operator and is the kernel's only
/// comptime parameter, so a whole work list is one dispatch.
fn run_sigma_p_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    tensor_rank: u32,
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
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($rank:expr) => {
                unsafe {
                    sigma_p_kernel::launch_unchecked::<f64, R>(
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
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SQRTPI,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $rank,
                        per_unit,
                    );
                }
            };
        }

        // Only `tensor_rank == 1` (int1e_sp) and 3 (int1e_sigma) exist; the
        // original per-pair launcher dispatched exactly this way.
        if tensor_rank == 1 {
            launch_with!(1u32);
        } else {
            launch_with!(3u32);
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched σ·p run.
fn dispatch_sigma_p_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    tensor_rank: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_sigma_p_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, tensor_rank)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_sigma_p_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, tensor_rank)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_sigma_p_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, tensor_rank)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_sigma_p_batches::<cubecl_hip::HipRuntime>(client, basis, groups, tensor_rank)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_sigma_p_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, tensor_rank)
        }
    }
}

/// One shell pair through the batched σ·p path — a one-pair group through the
/// same kernel a wide batch uses (Task 35-D).
///
/// Driven live by [`launch_int1e_sp_spinor_pair`] (the FND-05 path), which feeds
/// these gc blocks into `cart_to_spinor_si_2d`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sigma_p_on_backend(
    backend: &ResolvedBackend,
    tensor_rank: u32,
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
) -> Result<Vec<f64>, cintxRsError> {
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
        (tensor_rank * N_GC) as usize,
        sigma_p_g_per_axis(li as usize, lj as usize),
        1,
    );
    Ok(
        dispatch_sigma_p_batches(backend, &handles, std::slice::from_ref(&group), tensor_rank)
            .pop()
            .unwrap_or_default(),
    )
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / D-03 Wave 0): gauge-origin `x1i`-with-origin σ·p variant.
//
//  THE genuinely-new device math of Phase 30. Folds the gauge-origin GIAO factor
//  into the σ·p G-tensor for `int1e_cg_sa10sp` (the smallest COMMON_ORIG-reading
//  family — the Wave-0 micro-test vehicle).
//
//  The gauge factor is NOT a post-multiply cross-product (RESEARCH Pitfall 1). It
//  is the position recurrence `CINTx1i_1e` (g1e.c:446-448), folded through the
//  G-tensor build macro G1E_RCI:
//      x1i(g, origin)[i] = g[i+1] + origin * g[i]            // bra raise + shift
//  with origin = dri = ri − env[PTR_COMMON_ORIG] supplied per-axis from the host.
//
//  cg_sa10sp G-tensor build (intor3.c:1141-1160, common_factor *= 0.5):
//      g1 = G1E_D_J(g0)            ket nabla  ∇_j  (f[i] = j*g[i−dj] − 2aj*g[i+dj])
//      g2 = G1E_RCI(g0, dri)       x1i-with-origin
//      g3 = G1E_RCI(g1, dri)       x1i-with-origin of the ket-nabla
//  then the 9-element s[] product mix → 12-component gout (3 groups × gc 4-block).
//
//  Setting origin = [0,0,0] makes x1i(g,0)[i] = g[i+1] = G1E_R_I — i.e. the
//  `int1e_giao_sa10sp` natural-center build. Because the Wave-0 fixture places the
//  bra (i) shell at the coordinate origin, common_orig = [0,0,0] ⇒ dri = ri − 0 =
//  [0,0,0], so the cg result MUST collapse to giao — the live-gauge witness.
//
//  Headroom matches sigma_1e.rs::sigma_ov_kernel: nmax = li+lj+2 (x1i reads i+1,
//  composed needs +2 bra slack), lj_ext = lj+1 (∇_j reads j+1).
//  int1e_sp (tensor_rank==1) and its kernel above are NOT touched.
// ═════════════════════════════════════════════════════════════════════════════

/// `CINTnabla1j_1e` (g1e.c) ket nabla at index `idx0` (cart j-exponent `jexp`):
/// `f[i] = jexp*g[idx0−dj] − 2aj*g[idx0+dj]` (the `jexp==0` branch drops the
/// lowering term). Mirrors `sigma_1e.rs::nabla_j`.
#[cube]
fn sigma_p_nabla_j<F: Float>(g: &Array<F>, idx0: u32, dj: u32, aj2: F, jexp: u32) -> F {
    if jexp == 0u32 {
        aj2 * g[(idx0 + dj) as usize]
    } else {
        F::cast_from(jexp) * g[(idx0 - dj) as usize] + aj2 * g[(idx0 + dj) as usize]
    }
}

/// `CINTx1i_1e` (g1e.c:446-448) gauge `x1i`-with-origin at index `idx0`:
/// `g[idx0+1] + origin * g[idx0]` (bra raise by one + per-axis origin shift). This
/// is the gauge fold; NOT a cross-product.
#[cube]
fn sigma_p_x1i<F: Float>(g: &Array<F>, idx0: u32, origin: F) -> F {
    g[(idx0 + 1u32) as usize] + origin * g[idx0 as usize]
}

/// `x1i` of the ket-nabla `g1` (= `G1E_RCI(g3, g1)`): apply `∇_j` first at the two
/// bra-stencil points `idx0` and `idx0+1`, then the bra raise + origin shift.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sigma_p_x1i_of_j<F: Float>(g: &Array<F>, idx0: u32, dj: u32, aj2: F, jexp: u32, origin: F) -> F {
    let g1_i = sigma_p_nabla_j::<F>(g, idx0, dj, aj2, jexp);
    let g1_ip = sigma_p_nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
    g1_ip + origin * g1_i
}

/// Device kernel — `int1e_cg_sa10sp` gauge-origin σ·p assembler (rank 3, 12 gout
/// components → 3 groups × gc 4-block, component-leading KET-major).
///
/// `origin_x/y/z` carry the per-axis gauge origin `dri = ri − common_orig`. Single
/// `UNIT_POS==0` work item, mirroring `sigma_p_kernel`. `#[comptime] variant` is
/// reserved (`0 = cg_sa10sp`) so Wave 1 can switch the gout mix without forking.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_p_cg_sa10sp_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
    origin_x: F,
    origin_y: F,
    origin_z: F,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] _variant: u32,
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

            // Headroom: +2 in bra (x1i reads i+1; composed x1i-of-∇_j needs +2
            // slack), +1 in ket (∇_j reads j+1). Matches sigma_ov_kernel.
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
            // rank 3 → 12 gc blocks per (ci,cj).
            let rank = 3u32;
            let total_len = rank * N_GC * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
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

                    // ── OVERLAP base G-tensor g0 (fixed-center VRR + HRR). ──
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    sigma_p_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    if lj_ext >= 1u32 {
                        sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                    }

                    let aj2 = F::new(-2.0_f32) * aj;

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

                                            let bx = gx + jx * dj + ix;
                                            let by = gy + jy * dj + iy;
                                            let bz = gz + jz * dj + iz;

                                            // g0 = overlap base.
                                            let g0x = g[bx as usize];
                                            let g0y = g[by as usize];
                                            let g0z = g[bz as usize];
                                            // g1 = ∇_j(g0) (ket nabla, G1E_D_J).
                                            let g1x = sigma_p_nabla_j::<F>(g, bx, dj, aj2, jx);
                                            let g1y = sigma_p_nabla_j::<F>(g, by, dj, aj2, jy);
                                            let g1z = sigma_p_nabla_j::<F>(g, bz, dj, aj2, jz);
                                            // g2 = x1i(g0, dri) (gauge fold, G1E_RCI).
                                            let g2x = sigma_p_x1i::<F>(g, bx, origin_x);
                                            let g2y = sigma_p_x1i::<F>(g, by, origin_y);
                                            let g2z = sigma_p_x1i::<F>(g, bz, origin_z);
                                            // g3 = x1i(g1, dri) (gauge fold of ∇_j).
                                            let g3x =
                                                sigma_p_x1i_of_j::<F>(g, bx, dj, aj2, jx, origin_x);
                                            let g3y =
                                                sigma_p_x1i_of_j::<F>(g, by, dj, aj2, jy, origin_y);
                                            let g3z =
                                                sigma_p_x1i_of_j::<F>(g, bz, dj, aj2, jz, origin_z);

                                            // 9-element s[] product mix (intor3.c:1141).
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

                                            // 12-component gout → 3 groups × gc 4-block.
                                            // group 0 (gc_x..gc_1) = gout[0..4]:
                                            //   s8+s4, -s3, -s6, s7-s5
                                            let gp0 = base;
                                            gc_out[(gp0 + elem) as usize] += weight * (s8 + s4);
                                            gc_out[(gp0 + block_len + elem) as usize] +=
                                                weight * (-s3);
                                            gc_out[(gp0 + 2u32 * block_len + elem) as usize] +=
                                                weight * (-s6);
                                            gc_out[(gp0 + 3u32 * block_len + elem) as usize] +=
                                                weight * (s7 - s5);
                                            // group 1 = gout[4..8]:
                                            //   -s1, s0+s8, -s7, -s6+s2
                                            let gp1 = base + N_GC * block_len;
                                            gc_out[(gp1 + elem) as usize] += weight * (-s1);
                                            gc_out[(gp1 + block_len + elem) as usize] +=
                                                weight * (s0 + s8);
                                            gc_out[(gp1 + 2u32 * block_len + elem) as usize] +=
                                                weight * (-s7);
                                            gc_out[(gp1 + 3u32 * block_len + elem) as usize] +=
                                                weight * (-s6 + s2);
                                            // group 2 = gout[8..12]:
                                            //   -s2, -s5, s4+s0, s3-s1
                                            let gp2 = base + 2u32 * N_GC * block_len;
                                            gc_out[(gp2 + elem) as usize] += weight * (-s2);
                                            gc_out[(gp2 + block_len + elem) as usize] +=
                                                weight * (-s5);
                                            gc_out[(gp2 + 2u32 * block_len + elem) as usize] +=
                                                weight * (s4 + s0);
                                            gc_out[(gp2 + 3u32 * block_len + elem) as usize] +=
                                                weight * (s3 - s1);

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

/// `g_per_axis` for one gauge-origin σ·p class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn sigma_p_cg_g_per_axis(li: usize, lj: usize) -> usize {
    // +2 in bra, +1 in ket: nmax = li+lj+2, lj_ext = lj+1.
    (li + lj + 3) * (lj + 2)
}

/// Evaluate every launch group of a batched gauge-origin σ·p run (Task 35-D,
/// wave 5). `_variant` is fixed by the caller, so a work list is one dispatch.
fn run_sigma_p_cg_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin: [f64; 3],
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
            sigma_p_cg_sa10sp_kernel::launch_unchecked::<f64, R>(
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
                origin[0],
                origin[1],
                origin[2],
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                0u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched gauge-origin σ·p run.
fn dispatch_sigma_p_cg_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin: [f64; 3],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_sigma_p_cg_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, origin)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_sigma_p_cg_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, origin)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_sigma_p_cg_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, origin)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_sigma_p_cg_batches::<cubecl_hip::HipRuntime>(client, basis, groups, origin)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_sigma_p_cg_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, origin)
        }
    }
}

/// One shell pair through the batched gauge-origin σ·p path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sigma_p_cg_on_backend(
    backend: &ResolvedBackend,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    origin: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
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
        (3 * N_GC) as usize,
        sigma_p_cg_g_per_axis(li as usize, lj as usize),
        1,
    );
    Ok(
        dispatch_sigma_p_cg_batches(backend, &handles, std::slice::from_ref(&group), origin)
            .pop()
            .unwrap_or_default(),
    )
}

/// Live `int1e_cg_sa10sp` Spinor launcher — the Phase-30 Wave-0 gauge path.
///
/// Composes the gauge-origin σ·p assembler ([`run_sigma_p_cg_on_backend`], rank 3,
/// `common_factor *= 0.5`) with the imaginary-ket spin-included transform
/// [`cart_to_spinor_si_2di`] (`c2s_si_1ei`), producing the flat interleaved-complex
/// rank-3 spinor block for one shell pair `(i, j)`.
///
/// `origin` is the per-axis gauge shift `dri = ri − common_orig`; the host reads
/// `common_orig` from `plan.operator_env_params.common_orig.unwrap_or([0;3])`. With
/// `origin = [0,0,0]` the gauge fold collapses to the `int1e_giao_sa10sp`
/// natural-center build (the live-gauge witness — see kernel docs).
///
/// Output layout: 3 stacked spinor matrices, column-major (ket outer, bra inner),
/// interleaved complex, contraction-major:
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
///
/// Applies its OWN fail-closed staging guard BEFORE any write (Phase-28 CR-01).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_cg_sa10sp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    origin: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 3;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed staging guard BEFORE any write (full-block, ×2 complex, ×rank).
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // ── Device gauge σ·p assembler: 3*N_GC component-leading gc blocks per (ci,cj). ──
    let mut gc = run_sigma_p_cg_on_backend(
        backend,
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        origin,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
    )?;

    // s/p normalization scale × the cg_sa10sp common_factor (0.5, intor3.c:1163).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }

    let group_out = ni_sp * nj_sp * 2; // one spinor matrix's flat length
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // c2s_si_1ei owns the KET→BRA transpose (imaginary-ket arm).
                cart_to_spinor_si_2di::<F>(
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Live `int1e_giao_sa10sp` Spinor launcher — the natural-center gauge family.
///
/// `int1e_giao_sa10sp` shares the `int1e_cg_sa10sp` gout body byte-for-byte; the
/// sole difference is the G-tensor builder — `G1E_R_I` (natural bra center, no
/// gauge shift) instead of `G1E_RCI` (`dri = ri − common_orig`). In the
/// `CINTx1i_1e` recurrence `f[i] = g[i+1] + origin·g[i]`, `G1E_R_I` is exactly
/// `origin = [0,0,0]`. So this launcher is `launch_int1e_cg_sa10sp_spinor_pair`
/// with `origin = [0,0,0]` (verified by the Wave-0 cg→giao collapse witness:
/// `giao_sigma_micro` already proves cg at `common_orig=[0,0,0]` is byte-identical
/// to vendored `int1e_giao_sa10sp_spinor`).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_giao_sa10sp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    // giao = cg with the natural bra center: x1i origin = [0,0,0] (G1E_R_I).
    launch_int1e_cg_sa10sp_spinor_pair::<F>(
        backend,
        li,
        kappa_i,
        lj,
        kappa_j,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
        ri,
        rj,
        [0.0, 0.0, 0.0],
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        staging,
    )
}

/// Live `int1e_cg_sa10nucsp` / `int1e_giao_sa10nucsp` Spinor launcher (Sub-wave 1b)
/// — the NEW Rys+gauge **nuclear** engine, rank 3, c2s_si_1ei (imaginary ket).
///
/// Same shell-pair contract as `launch_int1e_cg_sa10sp_spinor_pair`, but the cart
/// blocks come from the Rys nuclear G-tensor with the gauge `x1i`-with-origin fold
/// applied INSIDE the Rys root loop (`super::sigma_1e_nuc::run_sigma_nuc_gauge_on_backend`).
/// `gauge` is `dri = ri − common_orig` (cg) or `[0,0,0]` (giao natural center).
/// `origin_coords`/`origin_charges` are the nuclear-attraction centers (charge `−Z`
/// per atom). The 12-component (3 tensor × gc 4-block) gout is byte-identical to the
/// cg/giao nucsp body (intor3.c:1230 / :1547); only the builder origin differs.
///
/// Output layout: 3 stacked spinor matrices, column-major (ket outer, bra inner),
/// interleaved complex, contraction-major:
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
///
/// Carries its OWN fail-closed full-block staging guard (`ni_sp*nj_sp*2*3`) before
/// any write; the Rys nroots fail-closed guard lives in `run_sigma_nuc_gauge_on_backend`
/// (never clamped — CR-01 / T-30-01b-03).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_sa10nucsp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    gauge: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 3;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed full-block staging guard BEFORE any write (Phase-28 CR-01,
    // T-30-01b-02). No partial guards.
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Rys+gauge nuclear cart gc blocks. The fail-closed nroots guard (never clamp,
    // T-30-01b-03) is inside run_sigma_nuc_gauge_on_backend.
    let mut gc = super::sigma_1e_nuc::run_sigma_nuc_gauge_on_backend(
        backend,
        li,
        lj,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
        ri,
        rj,
        gauge,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        origin_coords,
        origin_charges,
    )?;

    // s/p normalization × the cg/giao nucsp common_factor (0.5, intor3.c:1239).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }

    let group_out = ni_sp * nj_sp * 2;
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // c2s_si_1ei owns the KET→BRA transpose (imaginary-ket arm).
                cart_to_spinor_si_2di::<F>(
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / Sub-wave 1a): int1e_spgsp 8-G-tensor London overlap.
//
//  `<G SIGMA DOT P i| OVLP |SIGMA DOT P j>` — NOT the cg/giao gauge fold. This is
//  a distinct OVERLAP engine (no Rys atom-sum). Transcribed VERBATIM from
//  libcint autocode/intor3.c:1724-1758 (CINTgout1e_int1e_spgsp), including the
//  London `rirj = ri − rj` post-multiply.
//
//  THREE differences from cg_sa10sp:
//   (1) Builder is `G1E_R0I` (origin = `ri`, the bra center itself — NOT
//       `dri = ri − common_orig`; spgsp does NOT read PTR_COMMON_ORIG).
//   (2) An 8-G-tensor build (g0..g7):
//         g0 = overlap base
//         g1 = G1E_D_J(g0)          ket nabla ∇_j  (needs i_l+2 / j_l+1 headroom)
//         g2 = G1E_R0I(g0, ri)      x1i-with-origin=ri  (bra raise + ri shift)
//         g3 = G1E_R0I(g1, ri)      x1i-with-origin=ri of the ket-nabla
//         g4 = G1E_D_I(g0)          bra nabla ∇_i  (back-compose)
//         g5 = G1E_D_I(g1)
//         g6 = G1E_D_I(g2)
//         g7 = G1E_D_I(g3)
//       folding a 27-component cart mix s[0..26] into the 12-component gout.
//   (3) A London `rirj = ri − rj` (ri MINUS rj, NOT common_orig) post-multiply
//       in the gout: `c[k] = rirj[k]`, applied per-component (intor3.c:1742-1757).
//
//  The London/gauge fold is the `x1i` position recurrence (sigma_p_x1i), NOT a
//  cross-product (RESEARCH Pitfall 1). The 12-component gout maps to 3 groups ×
//  gc 4-block component-leading KET-major (k = tensor*4 + e1), the same layout the
//  cg_sa10sp kernel uses, consumed by the c2s_si_1ei (imaginary-ket) transform.
//
//  Headroom: nmax = li+lj+2 (D_J reads i+2 of g0; x1i reads i+1; D_I reads ±1),
//  lj_ext = lj+1 (D_J reads j+1). Identical to the cg kernel; int1e_sp and the
//  cg/giao paths above are NOT touched.
// ═════════════════════════════════════════════════════════════════════════════

/// `CINTnabla1i_1e` (g1e.c:322,345) bra nabla applied to a STENCIL value triple
/// `(v_lo, v_hi)` already read at bra indices `ix−1` and `ix+1`:
/// `f[ix] = ix*v_lo − 2ai*v_hi` (the `ix==0` branch drops the lowering term).
/// `ai2 = −2*ai`.
#[cube]
fn sigma_p_nabla_i_combine<F: Float>(v_lo: F, v_hi: F, ai2: F, iexp: u32) -> F {
    if iexp == 0u32 {
        ai2 * v_hi
    } else {
        F::cast_from(iexp) * v_lo + ai2 * v_hi
    }
}

/// Device kernel — `int1e_spgsp` 8-G-tensor London overlap assembler (rank 3, 12
/// gout components → 3 groups × gc 4-block, component-leading KET-major).
///
/// `rix/riy/riz` and `rjx/rjy/rjz` carry the bra/ket centers; the x1i origin is
/// `ri` (G1E_R0I) and the London phase is `rirj = ri − rj`. NEITHER reads
/// PTR_COMMON_ORIG. Single `UNIT_POS==0` work item, mirroring
/// `sigma_p_cg_sa10sp_kernel`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_p_spgsp_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
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

            // Headroom for the 8-G chain g7 = D_I(R0I(D_J(g0))): the deepest read
            // is D_I at bra i (reads i+1) of R0I (reads another +1) of D_J — so
            // the base G-tensor must be valid up to bra index li+2 at ket jx+1.
            // HRR fills g[j*dj+i] only for i <= nmax-j, so nmax must satisfy
            // (li+2) <= nmax-(lj+1) ⇒ nmax = li+lj+3 (ONE more bra level than the
            // cg kernel's li+lj+2). lj_ext = lj+1 (D_J reads j+1).
            let nmax = li + lj + 3u32;
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
            let rank = 3u32;
            let total_len = rank * N_GC * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // London phase rirj = ri − rj (NOT common_orig).
            let cx = rix - rjx;
            let cy = riy - rjy;
            let cz = riz - rjz;

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

                    // ── OVERLAP base G-tensor g0 (fixed-center VRR + HRR). ──
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    sigma_p_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    sigma_p_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    if lj_ext >= 1u32 {
                        sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                    }

                    let aj2 = F::new(-2.0_f32) * aj;
                    let ai2 = F::new(-2.0_f32) * ai;

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

                                            // Per-axis base index n = jcart*dj + icart.
                                            let bx = gx + jx * dj + ix;
                                            let by = gy + jy * dj + iy;
                                            let bz = gz + jz * dj + iz;

                                            // ── Per-axis g0..g7. The D_I (bra nabla)
                                            // back-compose of g0..g3 (→ g4..g7) reads the
                                            // bra-neighbors ix±1, so g0..g3 are evaluated
                                            // at ix−1, ix, ix+1 (x-axis) etc.

                                            // X axis ────────────────────────────────
                                            // g0 at ix-1/ix/ix+1
                                            let g0xm = if ix == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                g[(bx - 1u32) as usize]
                                            };
                                            let g0x = g[bx as usize];
                                            let g0xp = g[(bx + 1u32) as usize];
                                            // g1 = D_J(g0) at ix-1/ix/ix+1
                                            let g1xm = if ix == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_nabla_j::<F>(g, bx - 1u32, dj, aj2, jx)
                                            };
                                            let g1x = sigma_p_nabla_j::<F>(g, bx, dj, aj2, jx);
                                            let g1xp =
                                                sigma_p_nabla_j::<F>(g, bx + 1u32, dj, aj2, jx);
                                            // g2 = x1i(g0, ri) at ix-1/ix/ix+1
                                            let g2xm = if ix == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i::<F>(g, bx - 1u32, rix)
                                            };
                                            let g2x = sigma_p_x1i::<F>(g, bx, rix);
                                            let g2xp = sigma_p_x1i::<F>(g, bx + 1u32, rix);
                                            // g3 = x1i(g1, ri) at ix-1/ix/ix+1
                                            let g3xm = if ix == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i_of_j::<F>(
                                                    g,
                                                    bx - 1u32,
                                                    dj,
                                                    aj2,
                                                    jx,
                                                    rix,
                                                )
                                            };
                                            let g3x =
                                                sigma_p_x1i_of_j::<F>(g, bx, dj, aj2, jx, rix);
                                            let g3xp = sigma_p_x1i_of_j::<F>(
                                                g,
                                                bx + 1u32,
                                                dj,
                                                aj2,
                                                jx,
                                                rix,
                                            );
                                            // g4..g7 = D_I(g0..g3) at ix.
                                            let g4x =
                                                sigma_p_nabla_i_combine::<F>(g0xm, g0xp, ai2, ix);
                                            let g5x =
                                                sigma_p_nabla_i_combine::<F>(g1xm, g1xp, ai2, ix);
                                            let g6x =
                                                sigma_p_nabla_i_combine::<F>(g2xm, g2xp, ai2, ix);
                                            let g7x =
                                                sigma_p_nabla_i_combine::<F>(g3xm, g3xp, ai2, ix);

                                            // Y axis ────────────────────────────────
                                            let g0ym = if iy == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                g[(by - 1u32) as usize]
                                            };
                                            let g0y = g[by as usize];
                                            let g0yp = g[(by + 1u32) as usize];
                                            let g1ym = if iy == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_nabla_j::<F>(g, by - 1u32, dj, aj2, jy)
                                            };
                                            let g1y = sigma_p_nabla_j::<F>(g, by, dj, aj2, jy);
                                            let g1yp =
                                                sigma_p_nabla_j::<F>(g, by + 1u32, dj, aj2, jy);
                                            let g2ym = if iy == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i::<F>(g, by - 1u32, riy)
                                            };
                                            let g2y = sigma_p_x1i::<F>(g, by, riy);
                                            let g2yp = sigma_p_x1i::<F>(g, by + 1u32, riy);
                                            let g3ym = if iy == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i_of_j::<F>(
                                                    g,
                                                    by - 1u32,
                                                    dj,
                                                    aj2,
                                                    jy,
                                                    riy,
                                                )
                                            };
                                            let g3y =
                                                sigma_p_x1i_of_j::<F>(g, by, dj, aj2, jy, riy);
                                            let g3yp = sigma_p_x1i_of_j::<F>(
                                                g,
                                                by + 1u32,
                                                dj,
                                                aj2,
                                                jy,
                                                riy,
                                            );
                                            let g4y =
                                                sigma_p_nabla_i_combine::<F>(g0ym, g0yp, ai2, iy);
                                            let g5y =
                                                sigma_p_nabla_i_combine::<F>(g1ym, g1yp, ai2, iy);
                                            let g6y =
                                                sigma_p_nabla_i_combine::<F>(g2ym, g2yp, ai2, iy);
                                            let g7y =
                                                sigma_p_nabla_i_combine::<F>(g3ym, g3yp, ai2, iy);

                                            // Z axis ────────────────────────────────
                                            let g0zm = if iz == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                g[(bz - 1u32) as usize]
                                            };
                                            let g0z = g[bz as usize];
                                            let g0zp = g[(bz + 1u32) as usize];
                                            let g1zm = if iz == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_nabla_j::<F>(g, bz - 1u32, dj, aj2, jz)
                                            };
                                            let g1z = sigma_p_nabla_j::<F>(g, bz, dj, aj2, jz);
                                            let g1zp =
                                                sigma_p_nabla_j::<F>(g, bz + 1u32, dj, aj2, jz);
                                            let g2zm = if iz == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i::<F>(g, bz - 1u32, riz)
                                            };
                                            let g2z = sigma_p_x1i::<F>(g, bz, riz);
                                            let g2zp = sigma_p_x1i::<F>(g, bz + 1u32, riz);
                                            let g3zm = if iz == 0u32 {
                                                F::new(0.0_f32)
                                            } else {
                                                sigma_p_x1i_of_j::<F>(
                                                    g,
                                                    bz - 1u32,
                                                    dj,
                                                    aj2,
                                                    jz,
                                                    riz,
                                                )
                                            };
                                            let g3z =
                                                sigma_p_x1i_of_j::<F>(g, bz, dj, aj2, jz, riz);
                                            let g3zp = sigma_p_x1i_of_j::<F>(
                                                g,
                                                bz + 1u32,
                                                dj,
                                                aj2,
                                                jz,
                                                riz,
                                            );
                                            let g4z =
                                                sigma_p_nabla_i_combine::<F>(g0zm, g0zp, ai2, iz);
                                            let g5z =
                                                sigma_p_nabla_i_combine::<F>(g1zm, g1zp, ai2, iz);
                                            let g6z =
                                                sigma_p_nabla_i_combine::<F>(g2zm, g2zp, ai2, iz);
                                            let g7z =
                                                sigma_p_nabla_i_combine::<F>(g3zm, g3zp, ai2, iz);

                                            // ── 27-element s[] product mix (intor3.c:1742-1768). ──
                                            let s0 = g7x * g0y * g0z;
                                            let s1 = g6x * g1y * g0z;
                                            let s2 = g6x * g0y * g1z;
                                            let s3 = g5x * g2y * g0z;
                                            let s4 = g4x * g3y * g0z;
                                            let s5 = g4x * g2y * g1z;
                                            let s6 = g5x * g0y * g2z;
                                            let s7 = g4x * g1y * g2z;
                                            let s8 = g4x * g0y * g3z;
                                            let s9 = g3x * g4y * g0z;
                                            let s10 = g2x * g5y * g0z;
                                            let s11 = g2x * g4y * g1z;
                                            let s12 = g1x * g6y * g0z;
                                            let s13 = g0x * g7y * g0z;
                                            let s14 = g0x * g6y * g1z;
                                            let s15 = g1x * g4y * g2z;
                                            let s16 = g0x * g5y * g2z;
                                            let s17 = g0x * g4y * g3z;
                                            let s18 = g3x * g0y * g4z;
                                            let s19 = g2x * g1y * g4z;
                                            let s20 = g2x * g0y * g5z;
                                            let s21 = g1x * g2y * g4z;
                                            let s22 = g0x * g3y * g4z;
                                            let s23 = g0x * g2y * g5z;
                                            let s24 = g1x * g0y * g6z;
                                            let s25 = g0x * g1y * g6z;
                                            let s26 = g0x * g0y * g7z;
                                            // Silence unused-by-gout cart products
                                            // (s5/s7/s11/s15/s19/s21 — the spgsp London
                                            // mix uses only the other 21 of the 27).
                                            let _ = s5 + s7 + s11 + s15 + s19 + s21;

                                            let elem = cj_idx * nci + ci_idx;

                                            // 12-component gout (intor3.c:1742-1757) →
                                            // 3 groups × gc 4-block (k = tensor*4 + e1).
                                            let gp0 = base;
                                            // gout[0] = c1*s17 - c2*s14 - c1*s25 + c2*s22
                                            gc_out[(gp0 + elem) as usize] += weight
                                                * (cy * s17 - cz * s14 - cy * s25 + cz * s22);
                                            // gout[1] = c1*s24 - c1*s8
                                            gc_out[(gp0 + block_len + elem) as usize] +=
                                                weight * (cy * s24 - cy * s8);
                                            // gout[2] = -c2*s4 + c2*s12
                                            gc_out[(gp0 + 2u32 * block_len + elem) as usize] +=
                                                weight * (-cz * s4 + cz * s12);
                                            // gout[3] = c1*s6 - c2*s3 + c1*s16 - c2*s13 + c1*s26 - c2*s23
                                            gc_out[(gp0 + 3u32 * block_len + elem) as usize] +=
                                                weight
                                                    * (cy * s6 - cz * s3 + cy * s16 - cz * s13
                                                        + cy * s26
                                                        - cz * s23);

                                            let gp1 = base + N_GC * block_len;
                                            // gout[4] = -c0*s17 + c0*s25
                                            gc_out[(gp1 + elem) as usize] +=
                                                weight * (-cx * s17 + cx * s25);
                                            // gout[5] = c2*s18 - c0*s24 - c2*s2 + c0*s8
                                            gc_out[(gp1 + block_len + elem) as usize] +=
                                                weight * (cz * s18 - cx * s24 - cz * s2 + cx * s8);
                                            // gout[6] = c2*s1 - c2*s9
                                            gc_out[(gp1 + 2u32 * block_len + elem) as usize] +=
                                                weight * (cz * s1 - cz * s9);
                                            // gout[7] = c2*s0 - c0*s6 + c2*s10 - c0*s16 + c2*s20 - c0*s26
                                            gc_out[(gp1 + 3u32 * block_len + elem) as usize] +=
                                                weight
                                                    * (cz * s0 - cx * s6 + cz * s10 - cx * s16
                                                        + cz * s20
                                                        - cx * s26);

                                            let gp2 = base + 2u32 * N_GC * block_len;
                                            // gout[8] = c0*s14 - c0*s22
                                            gc_out[(gp2 + elem) as usize] +=
                                                weight * (cx * s14 - cx * s22);
                                            // gout[9] = -c1*s18 + c1*s2
                                            gc_out[(gp2 + block_len + elem) as usize] +=
                                                weight * (-cy * s18 + cy * s2);
                                            // gout[10] = c0*s4 - c1*s1 - c0*s12 + c1*s9
                                            gc_out[(gp2 + 2u32 * block_len + elem) as usize] +=
                                                weight * (cx * s4 - cy * s1 - cx * s12 + cy * s9);
                                            // gout[11] = c0*s3 - c1*s0 + c0*s13 - c1*s10 + c0*s23 - c1*s20
                                            gc_out[(gp2 + 3u32 * block_len + elem) as usize] +=
                                                weight
                                                    * (cx * s3 - cy * s0 + cx * s13 - cy * s10
                                                        + cx * s23
                                                        - cy * s20);

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

/// `g_per_axis` for one spgsp London class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn sigma_p_spgsp_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj+3, lj_ext = lj+1.
    (li + lj + 4) * (lj + 2)
}

/// Evaluate every launch group of a batched spgsp run (Task 35-D, wave 5).
///
/// The kernel has no comptime shape parameter, so a work list is one dispatch.
fn run_sigma_p_spgsp_batches<R: Runtime>(
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
            sigma_p_spgsp_kernel::launch_unchecked::<f64, R>(
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

/// Backend dispatch for a whole batched spgsp run.
fn dispatch_sigma_p_spgsp_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_sigma_p_spgsp_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_sigma_p_spgsp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_sigma_p_spgsp_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_sigma_p_spgsp_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_sigma_p_spgsp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched spgsp path — a one-pair group through the
/// same kernel a wide batch uses (Task 35-D).
///
/// The x1i origin is `ri` and the London phase is `ri − rj`; NEITHER reads
/// PTR_COMMON_ORIG.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sigma_p_spgsp_on_backend(
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
) -> Result<Vec<f64>, cintxRsError> {
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
        (3 * N_GC) as usize,
        sigma_p_spgsp_g_per_axis(li as usize, lj as usize),
        1,
    );
    Ok(
        dispatch_sigma_p_spgsp_batches(backend, &handles, std::slice::from_ref(&group))
            .pop()
            .unwrap_or_default(),
    )
}

/// Live `int1e_spgsp` Spinor launcher — the Sub-wave 1a 8-G London overlap path.
///
/// Composes the spgsp London overlap assembler ([`run_sigma_p_spgsp_on_backend`],
/// rank 3) with the imaginary-ket spin-included transform
/// [`cart_to_spinor_si_2di`] (`c2s_si_1ei`), producing the flat interleaved-complex
/// rank-3 spinor block for one shell pair `(i, j)`.
///
/// The x1i origin is `ri` (G1E_R0I) and the London phase is `ri − rj`; spgsp does
/// NOT read PTR_COMMON_ORIG, so no `origin`/`common_orig` argument is threaded.
///
/// Output layout: 3 stacked spinor matrices, column-major (ket outer, bra inner),
/// interleaved complex, contraction-major:
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
///
/// Applies its OWN fail-closed staging guard BEFORE any write (Phase-28 CR-01).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_spgsp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 3;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed staging guard BEFORE any write (full-block, ×2 complex, ×rank).
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    let mut gc = run_sigma_p_spgsp_on_backend(
        backend,
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
    )?;

    // s/p normalization scale (no extra common_factor for spgsp; ng[6]=1).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in gc.iter_mut() {
            *v *= sp_scale;
        }
    }

    let group_out = ni_sp * nj_sp * 2; // one spinor matrix's flat length
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // c2s_si_1ei owns the KET→BRA transpose (imaginary-ket arm).
                cart_to_spinor_si_2di::<F>(
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

/// libcint `CINTcommon_fac_sp` s/p normalization factor (matches
/// `one_electron.rs::common_fac_sp`). libcint moves the s/p spherical
/// normalization out of the c2s tables and into the primitive loop
/// (`g1e.c:120`), so the σ·p overlap base G-tensor (built with 1.0 for s/p in
/// the c2s convention) must be scaled by `common_fac_sp(li)*common_fac_sp(lj)`
/// before the spinor transform — exactly as the scalar/gradient 1e arms do.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64, // 1/(2*sqrt(pi))
        1 => 0.488602511902919921_f64, // sqrt(3/(4*pi))
        _ => 1.0,
    }
}

/// Live `int1e_sp` Spinor launcher — the FND-05 end-to-end path (Plan 28-04).
///
/// Composes the device σ·p assembler ([`run_sigma_p_on_backend`], `tensor_rank=1`)
/// with the host spin-included transform [`cart_to_spinor_si_2d`], producing the
/// flat interleaved-complex spinor block for one shell pair `(i, j)`.
///
/// # Layout
/// Output is column-major (ket-spinor outer, bra-spinor inner), interleaved
/// complex `[re,im,…]`, contraction-major:
/// `out[(j_global*ni_sp + i_global)*2 + {0:re,1:im}]` where
/// `ni_sp = nctr_i*spinor_len(li,kappa_i)`, `nj_sp = nctr_j*spinor_len(lj,kappa_j)`.
///
/// # nctr>1 (D-08 carryover)
/// Handles general contraction: the σ·p assembler emits one component-leading
/// gc 4-block per `(ci,cj)` contraction pair; this launcher slices each pair's
/// four KET-major gc blocks and folds them through `cart_to_spinor_si_2d` (which
/// owns the KET→BRA transpose internally — Pitfall 4 / Phase-27 D-06), scattering
/// the `di*dj*2` spinor sub-block into the contraction-major grid. NO second
/// transpose is applied in the launcher.
///
/// `coeff_i` / `coeff_j` are ROW-major `[ip*nctr + ic]` (the cintx `Shell`
/// convention after raw.rs transposes libcint's COLUMN-major env block — see
/// `project_raw_nctr_coeff_transpose`); the σ·p kernel reads them as
/// `coeff[pi*nctr_i + ci]`, matching that row-major layout.
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_sp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    // tensor_rank=1 → N_GC=4 gc blocks per (ci,cj). total_len matches the
    // assembler's per-(ci,cj) extent.
    let total_len = (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed BEFORE any write (OOM-safe stop, no partial writes).
    let staging_required = ni_sp * nj_sp * 2;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // ── Device σ·p assembler: 4 component-leading gc blocks per (ci,cj). ──
    let mut gc = run_sigma_p_on_backend(
        backend,
        1, // tensor_rank = 1 (int1e_sp)
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
    )?;

    // Apply the s/p normalization scale (matches the scalar/gradient 1e arms).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in gc.iter_mut() {
            *v *= sp_scale;
        }
    }

    // ── Per-(ci,cj): slice the 4 KET-major gc blocks, fold + scatter. ──
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            let gc_x = &gc[base..base + block_len];
            let gc_y = &gc[base + block_len..base + 2 * block_len];
            let gc_z = &gc[base + 2 * block_len..base + 3 * block_len];
            let gc_1 = &gc[base + 3 * block_len..base + 4 * block_len];

            // cart_to_spinor_si_2d owns the KET→BRA transpose; pass the gc blocks
            // as the assembler emits them (KET-major) — no launcher transpose.
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

            // Scatter the di*dj*2 spinor sub-block into the contraction-major grid.
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

    Ok(())
}

/// Pure-Rust host reference replicating [`sigma_p_kernel`] exactly.
///
/// Produces the same `tensor_rank * N_GC`-block component-leading gc accumulator
/// as the device kernel, for the device-vs-host parity test. Single shell pair,
/// `norm_i = norm_j = 1.0` (matching the in-kernel pdata recompute).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn sigma_p_host(
    tensor_rank: u32,
    li: u32,
    lj: u32,
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax = li_u + lj_u + 1;
    let lj_ext = lj_u;
    let dj = nmax + 1;
    let g_per_axis = (nmax + 1) * (lj_ext + 1);
    let gx = 0usize;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;
    let total_g = 3 * g_per_axis;

    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let block_len = nci * ncj;
    let n_blocks = (tensor_rank as usize) * (N_GC as usize);
    let total_len = n_blocks * block_len;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * total_len;
    let mut out = vec![0.0_f64; out_len];

    let pi_const = std::f64::consts::PI;

    for (pi, &ai) in exps_i.iter().enumerate() {
        for (pj, &aj) in exps_j.iter().enumerate() {
            let zeta = ai + aj;
            let aij2 = 0.5 / zeta;
            let rirjx = ri[0] - rj[0];
            let rirjy = ri[1] - rj[1];
            let rirjz = ri[2] - rj[2];
            let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
            let fac = (-ai * aj / zeta * rr).exp();
            let px = (ai * ri[0] + aj * rj[0]) / zeta;
            let py = (ai * ri[1] + aj * rj[1]) / zeta;
            let pz = (ai * ri[2] + aj * rj[2]) / zeta;

            let mut g = vec![0.0_f64; total_g];
            g[gx] = 1.0;
            g[gy] = 1.0;
            g[gz] = fac * SQRTPI * pi_const / (zeta * zeta.sqrt());

            let vrr_axis = |g: &mut [f64], base: usize, rijrx: f64| {
                if nmax >= 1 {
                    g[base + 1] = rijrx * g[base];
                    for n in 1..nmax {
                        g[base + n + 1] = (n as f64) * aij2 * g[base + n - 1] + rijrx * g[base + n];
                    }
                }
            };
            vrr_axis(&mut g, gx, px - ri[0]);
            vrr_axis(&mut g, gy, py - ri[1]);
            vrr_axis(&mut g, gz, pz - ri[2]);

            if lj_ext >= 1 {
                let hrr_axis = |g: &mut [f64], base: usize, rirj: f64| {
                    for j in 1..=lj_ext {
                        let i_max = nmax - j;
                        for i in 0..=i_max {
                            let idx_out = base + j * dj + i;
                            let idx_hi = base + (j - 1) * dj + (i + 1);
                            let idx_lo = base + (j - 1) * dj + i;
                            g[idx_out] = g[idx_hi] + rirj * g[idx_lo];
                        }
                    }
                };
                hrr_axis(&mut g, gx, rirjx);
                hrr_axis(&mut g, gy, rirjy);
                hrr_axis(&mut g, gz, rirjz);
            }

            // Bra nabla1i → g1.
            let mut g1 = vec![0.0_f64; total_g];
            let ai2 = -2.0 * ai;
            for axisn in 0..3 {
                let off = axisn * g_per_axis;
                for jn in 0..=lj_ext {
                    let jbase = jn * dj;
                    g1[off + jbase] = ai2 * g[off + jbase + 1];
                    for ix in 1..=li_u {
                        g1[off + jbase + ix] =
                            (ix as f64) * g[off + jbase + ix - 1] + ai2 * g[off + jbase + ix + 1];
                    }
                }
            }

            // Contract.
            for ci in 0..(nctr_i as usize) {
                let coeff_i_val = coeff_i[pi * (nctr_i as usize) + ci];
                for cj in 0..(nctr_j as usize) {
                    let coeff_j_val = coeff_j[pj * (nctr_j as usize) + cj];
                    let weight = coeff_i_val * coeff_j_val;
                    let base = (ci * (nctr_j as usize) + cj) * total_len;

                    let mut cj_idx = 0usize;
                    for ja in 0..=lj_u {
                        let jx = lj_u - ja;
                        let lj_minus_jx = lj_u - jx;
                        for jb in 0..=lj_minus_jx {
                            let jy = lj_minus_jx - jb;
                            let jz = lj_u - jx - jy;

                            let mut ci_idx = 0usize;
                            for ia in 0..=li_u {
                                let ix = li_u - ia;
                                let li_minus_ix = li_u - ix;
                                for ib in 0..=li_minus_ix {
                                    let iy = li_minus_ix - ib;
                                    let iz = li_u - ix - iy;

                                    let nx = jx * dj + ix;
                                    let ny = jy * dj + iy;
                                    let nz = jz * dj + iz;

                                    let g0x = g[gx + nx];
                                    let g0y = g[gy + ny];
                                    let g0z = g[gz + nz];
                                    let g1x = g1[gx + nx];
                                    let g1y = g1[gy + ny];
                                    let g1z = g1[gz + nz];

                                    let s0 = g1x * g0y * g0z;
                                    let s1 = g0x * g1y * g0z;
                                    let s2 = g0x * g0y * g1z;

                                    let elem = cj_idx * nci + ci_idx;
                                    out[base + elem] += weight * s0; // gc_x block 0
                                    out[base + block_len + elem] += weight * s1; // gc_y block 1
                                    out[base + 2 * block_len + elem] += weight * s2; // gc_z block 2
                                    // gc_1 scalar slot (block 3) stays 0.

                                    ci_idx += 1;
                                }
                            }
                            cj_idx += 1;
                        }
                    }
                }
            }
        }
    }

    out
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / Sub-wave 1c): int1e_cg_sa10sa01 / int1e_giao_sa10sa01
//  — the Rys+gauge rank-9 engine class.
//
//  <SIGMA CROSS RC i | SIGMA CROSS NABLA-RINV | j>. The highest-rank Wave-1 1e
//  engine: a rinv (int1e_type 1, single-center Rys) nuclear base, the both-side
//  nabla g1 = nabla_j(g0) + nabla_i(g0), the x1i-with-origin gauge fold
//  g2 = x1i(g0, o), g3 = x1i(g1, o), and a 36-component (9 tensor x 4 gc) gout
//  transcribed VERBATIM from autocode/intor3.c:998 (cg) / :1323 (giao).
//
//  The cg and giao gout BODIES are byte-for-byte identical; only the x1i origin
//  differs (G2E_RCI dri = ri - common_orig for cg; G2E_R_I ri for giao —
//  g2e.h:98/104). common_factor *= 0.5 (intor3.c:1117).
//
//  Output is routed through the REAL si transform cart_to_spinor_si_2d
//  (c2s_si_1e), NOT the imaginary si_2di the sp/nucsp arms use (Pitfall 2).
//
//  Rys roots are device-side, comptime-dispatched 1..=5, fail-closed above the
//  envelope (no clamp). nmax = li+lj+2, lj_ext = lj+1 (x1i reads i+1; the
//  composed x1i-of-(nabla_j+nabla_i) needs +2 bra / +1 ket slack). The overlap
//  HRR is reused (sigma_p_hrr_axis); the base VRR is root-dependent (nuclear),
//  so a local sa01_nuc_vrr_axis mirrors sigma_1e_nuc::nuc_vrr_axis.
// ═════════════════════════════════════════════════════════════════════════════

/// PIE4 = pi/4 (Rys weight normalization, matches `sigma_1e_nuc::PIE4`).
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const SA01_PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum on-device Rys root count (comptime arms 1..=5). Above this the kernel
/// would silently truncate the quadrature, so the launcher fails closed.
const SA01_MAX_DEVICE_NROOTS: u32 = 5;

/// Per-axis 2e nuclear VRR (root-dependent c00/b10; matches
/// `sigma_1e_nuc::nuc_vrr_axis`). Used by the rank-9 Rys+gauge sa01 kernel.
#[cube]
fn sa01_nuc_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, c00: F, b10: F, nmax: u32) {
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

/// `nabla_i(g)` bra-nabla at index `idx0` (cart i-exponent `iexp`):
/// `i*g[idx0-1] + ai2*g[idx0+1]` (matches `sigma_1e_nuc::nuc_nabla_i`).
#[cube]
fn sa01_nabla_i<F: Float>(g: &Array<F>, idx0: u32, ai2: F, iexp: u32) -> F {
    if iexp == 0u32 {
        ai2 * g[(idx0 + 1u32) as usize]
    } else {
        F::cast_from(iexp) * g[(idx0 - 1u32) as usize] + ai2 * g[(idx0 + 1u32) as usize]
    }
}

/// `g1 = nabla_j(g0) + nabla_i(g0)` (both-side nabla) at index `idx0` (cart
/// exponents `iexp`/`jexp`). Reuses `sigma_p_nabla_j` for the ket-side nabla_j.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sa01_g1_bothside<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    ai2: F,
    aj2: F,
    iexp: u32,
    jexp: u32,
) -> F {
    sigma_p_nabla_j::<F>(g, idx0, dj, aj2, jexp) + sa01_nabla_i::<F>(g, idx0, ai2, iexp)
}

/// `x1i(g1, origin)` where `g1 = nabla_j + nabla_i` (= `G2E_RCI(g3, g1)`): form g1
/// at the two bra-stencil points `idx0` (i-exponent `iexp`) and `idx0+1`
/// (i-exponent `iexp+1`, since the x1i raise consumes one bra level), then apply
/// the x1i shift `f[i] = g1[i+1] + origin*g1[i]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sa01_x1i_of_g1<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    ai2: F,
    aj2: F,
    iexp: u32,
    jexp: u32,
    origin: F,
) -> F {
    let g1_i = sa01_g1_bothside::<F>(g, idx0, dj, ai2, aj2, iexp, jexp);
    let g1_ip = sa01_g1_bothside::<F>(g, idx0 + 1u32, dj, ai2, aj2, iexp + 1u32, jexp);
    g1_ip + origin * g1_i
}

/// Rank-9 sa01 gout (VERBATIM `intor3.c:998`/`:1323`, 9 tensor x 4 gc = 36 comp).
/// Reads g0..g3 per axis, scatters into the 9-group x N_GC layout
/// (`gc_out[base + (group*N_GC + gc_block)*block_len + elem]`). cg/giao share
/// this body. `s[0..8]` per intor3.c:1022-1030.
#[cube]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn sa01_gout<F: Float>(
    gc_out: &mut Array<F>,
    base: u32,
    block_len: u32,
    elem: u32,
    weight: F,
    g0x: F,
    g0y: F,
    g0z: F,
    g1x: F,
    g1y: F,
    g1z: F,
    g2x: F,
    g2y: F,
    g2z: F,
    g3x: F,
    g3y: F,
    g3z: F,
) {
    let s0 = g3x * g0y * g0z;
    let s1 = g2x * g1y * g0z;
    let s2 = g2x * g0y * g1z;
    let s3 = g1x * g2y * g0z;
    let s4 = g0x * g3y * g0z;
    let s5 = g0x * g2y * g1z;
    let s6 = g1x * g0y * g2z;
    let s7 = g0x * g1y * g2z;
    let s8 = g0x * g0y * g3z;

    let nb = N_GC * block_len;
    // group 0: gout 0..3 = (-s7+s5, 0, 0, s8+s4)
    gc_out[(base + 0u32 * nb + 0u32 * block_len + elem) as usize] += weight * (-s7 + s5);
    gc_out[(base + 0u32 * nb + 3u32 * block_len + elem) as usize] += weight * (s8 + s4);
    // group 1: gout 4..7 = (s6, s5, s8, -s3)
    gc_out[(base + 1u32 * nb + 0u32 * block_len + elem) as usize] += weight * s6;
    gc_out[(base + 1u32 * nb + 1u32 * block_len + elem) as usize] += weight * s5;
    gc_out[(base + 1u32 * nb + 2u32 * block_len + elem) as usize] += weight * s8;
    gc_out[(base + 1u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s3);
    // group 2: gout 8..11 = (-s3, -s4, -s7, -s6)
    gc_out[(base + 2u32 * nb + 0u32 * block_len + elem) as usize] += weight * (-s3);
    gc_out[(base + 2u32 * nb + 1u32 * block_len + elem) as usize] += weight * (-s4);
    gc_out[(base + 2u32 * nb + 2u32 * block_len + elem) as usize] += weight * (-s7);
    gc_out[(base + 2u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s6);
    // group 3: gout 12..15 = (-s2, -s7, -s8, -s1)
    gc_out[(base + 3u32 * nb + 0u32 * block_len + elem) as usize] += weight * (-s2);
    gc_out[(base + 3u32 * nb + 1u32 * block_len + elem) as usize] += weight * (-s7);
    gc_out[(base + 3u32 * nb + 2u32 * block_len + elem) as usize] += weight * (-s8);
    gc_out[(base + 3u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s1);
    // group 4: gout 16..19 = (0, -s2+s6, 0, s8+s0)
    gc_out[(base + 4u32 * nb + 1u32 * block_len + elem) as usize] += weight * (-s2 + s6);
    gc_out[(base + 4u32 * nb + 3u32 * block_len + elem) as usize] += weight * (s8 + s0);
    // group 5: gout 20..23 = (s0, s1, s6, -s7)
    gc_out[(base + 5u32 * nb + 0u32 * block_len + elem) as usize] += weight * s0;
    gc_out[(base + 5u32 * nb + 1u32 * block_len + elem) as usize] += weight * s1;
    gc_out[(base + 5u32 * nb + 2u32 * block_len + elem) as usize] += weight * s6;
    gc_out[(base + 5u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s7);
    // group 6: gout 24..27 = (s1, s4, s5, -s2)
    gc_out[(base + 6u32 * nb + 0u32 * block_len + elem) as usize] += weight * s1;
    gc_out[(base + 6u32 * nb + 1u32 * block_len + elem) as usize] += weight * s4;
    gc_out[(base + 6u32 * nb + 2u32 * block_len + elem) as usize] += weight * s5;
    gc_out[(base + 6u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s2);
    // group 7: gout 28..31 = (-s0, -s3, -s2, -s5)
    gc_out[(base + 7u32 * nb + 0u32 * block_len + elem) as usize] += weight * (-s0);
    gc_out[(base + 7u32 * nb + 1u32 * block_len + elem) as usize] += weight * (-s3);
    gc_out[(base + 7u32 * nb + 2u32 * block_len + elem) as usize] += weight * (-s2);
    gc_out[(base + 7u32 * nb + 3u32 * block_len + elem) as usize] += weight * (-s5);
    // group 8: gout 32..35 = (0, 0, -s3+s1, s4+s0)
    gc_out[(base + 8u32 * nb + 2u32 * block_len + elem) as usize] += weight * (-s3 + s1);
    gc_out[(base + 8u32 * nb + 3u32 * block_len + elem) as usize] += weight * (s4 + s0);
}

/// Device kernel — the rank-9 Rys+gauge sa01 assembler (rinv nuclear base, single
/// center charge +1). `origin_{x,y,z}` is the x1i gauge shift (dri for cg, ri for
/// giao). 9 sigma-groups x N_GC blocks per (ci,cj). nroots is comptime.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sa01_rys_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
    rcx: F,
    rcy: F,
    rcz: F,
    origin_x: F,
    origin_y: F,
    origin_z: F,
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
            let nrys = nroots;
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
            // rank 9 (9 sigma-groups), N_GC=4 blocks each.
            let total_len = 9u32 * N_GC * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
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

                    // Single rinv center (charge +1).
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

                    let fac1 = F::new(2.0_f32) * pi_const * fac / zeta;

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

                        sa01_nuc_vrr_axis::<F>(g, gx, c00x, rt, nmax);
                        sa01_nuc_vrr_axis::<F>(g, gy, c00y, rt, nmax);
                        sa01_nuc_vrr_axis::<F>(g, gz, c00z, rt, nmax);
                        if lj_ext >= 1u32 {
                            sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                            sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                            sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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

                                                let bx = gx + jx * dj + ix;
                                                let by = gy + jy * dj + iy;
                                                let bz = gz + jz * dj + iz;

                                                let g0x = g[bx as usize];
                                                let g0y = g[by as usize];
                                                let g0z = g[bz as usize];

                                                // g1 = nabla_j(g0) + nabla_i(g0).
                                                let g1x = sa01_g1_bothside::<F>(
                                                    g, bx, dj, ai2, aj2, ix, jx,
                                                );
                                                let g1y = sa01_g1_bothside::<F>(
                                                    g, by, dj, ai2, aj2, iy, jy,
                                                );
                                                let g1z = sa01_g1_bothside::<F>(
                                                    g, bz, dj, ai2, aj2, iz, jz,
                                                );
                                                // g2 = x1i(g0, origin) (gauge fold).
                                                let g2x = sigma_p_x1i::<F>(g, bx, origin_x);
                                                let g2y = sigma_p_x1i::<F>(g, by, origin_y);
                                                let g2z = sigma_p_x1i::<F>(g, bz, origin_z);
                                                // g3 = x1i(g1, origin) (gauge fold of g1).
                                                let g3x = sa01_x1i_of_g1::<F>(
                                                    g, bx, dj, ai2, aj2, ix, jx, origin_x,
                                                );
                                                let g3y = sa01_x1i_of_g1::<F>(
                                                    g, by, dj, ai2, aj2, iy, jy, origin_y,
                                                );
                                                let g3z = sa01_x1i_of_g1::<F>(
                                                    g, bz, dj, ai2, aj2, iz, jz, origin_z,
                                                );

                                                sa01_gout::<F>(
                                                    gc_out,
                                                    base,
                                                    block_len,
                                                    cj_idx * nci + ci_idx,
                                                    weight,
                                                    g0x,
                                                    g0y,
                                                    g0z,
                                                    g1x,
                                                    g1y,
                                                    g1z,
                                                    g2x,
                                                    g2y,
                                                    g2z,
                                                    g3x,
                                                    g3y,
                                                    g3z,
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
                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// Evaluate every launch group of a batched `sa01_rys` run (Task 35-D, wave 5).
///
/// One dispatch per Rys order; the rinv centre and the gauge origin are
/// per-call scalars, so they stay kernel arguments rather than joining the
/// pair table.
fn run_sa01_rys_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rinv: [f64; 3],
    origin: [f64; 3],
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
                    sa01_rys_kernel::launch_unchecked::<f64, R>(
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
                        rinv[0],
                        rinv[1],
                        rinv[2],
                        origin[0],
                        origin[1],
                        origin[2],
                        SA01_PIE4,
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

/// Backend dispatch for a whole batched `sa01_rys` run.
fn dispatch_sa01_rys_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rinv: [f64; 3],
    origin: [f64; 3],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_sa01_rys_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, rinv, origin)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_sa01_rys_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rinv, origin)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_sa01_rys_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, rinv, origin)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_sa01_rys_batches::<cubecl_hip::HipRuntime>(client, basis, groups, rinv, origin)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_sa01_rys_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rinv, origin)
        }
    }
}

/// `g_per_axis` for one `sa01_rys` class.
fn sa01_rys_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj+2, lj_ext = lj+1.
    (li + lj + 3) * (lj + 2)
}

/// One shell pair through the batched `sa01_rys` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_sa01_rys_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u8,
    lj: u8,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rinv: [f64; 3],
    origin: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        (9 * N_GC) as usize,
        sa01_rys_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    Ok(dispatch_sa01_rys_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        rinv,
        origin,
    )
    .pop()
    .unwrap_or_default())
}

/// 30-01c-DEBUG cart-path discriminator: return the PRE-TRANSFORM cart `gc`
/// buffer (the `run_sa01_rys_on_backend` output × the s/p × 0.5 common_factor
/// scale), exactly what `launch_int1e_sa10sa01_spinor_pair` feeds into
/// `cart_to_spinor_si_2d`. Bypasses the c2s_si transform so a vendor cart
/// comparison isolates the g-tensor assembly from the spinor transform.
///
/// Layout (per (ci,cj) contraction block, then 9 groups × 4 gc-blocks (x,y,z,1)
/// × `ncart_i*ncart_j`): identical to the `gc` consumed at sigma_p.rs:2810-2816.
/// `origin` is the x1i gauge shift (dri for cg, ri for giao); `rinv` is the rinv
/// center. NOT a production path — debugging instrument only.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn sa01_cart_gc_for_test(
    backend: &ResolvedBackend,
    li: u8,
    lj: u8,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rinv: [f64; 3],
    origin: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let order = li as usize + lj as usize + 2;
    let nroots = (order / 2 + 1) as u32;
    if nroots > SA01_MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }
    let mut gc = run_sa01_rys_on_backend(
        backend, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, rinv, origin, exps_i,
        exps_j, coeff_i, coeff_j,
    )?;
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }
    Ok(gc)
}

/// Live `int1e_cg_sa10sa01` / `int1e_giao_sa10sa01` Spinor launcher (rank 9).
///
/// Builds the rank-9 Rys+gauge cart gc blocks (9 groups x 4 gc), applies the s/p
/// normalization x the sa01 common_factor (0.5, intor3.c:1117), folds each of the
/// 9 sigma-groups through the REAL `cart_to_spinor_si_2d` (c2s_si_1e — NOT the
/// imaginary si_2di), and scatters into the contraction-major spinor grid.
///
/// `origin` is the x1i gauge shift: `dri = ri - common_orig` for cg, `ri` for
/// giao (the caller threads it). `rinv` is the rinv center (PTR_RINV_ORIG default
/// [0,0,0]). Carries its OWN fail-closed full-block (x9) staging guard plus a
/// fail-closed `nroots > MAX_DEVICE_NROOTS -> UnsupportedApi` guard (no clamp).
///
/// Output layout: 9 stacked spinor matrices, column-major (ket outer, bra inner),
/// interleaved complex, contraction-major:
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_sa10sa01_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rinv: [f64; 3],
    origin: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 9;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed full-block (x9) staging guard BEFORE any write (Phase-28 CR-01,
    // T-30-01c-03). No partial guards.
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Fail-closed Rys-nroots guard (T-30-01c-04): never clamp.
    let order = li as usize + lj as usize + 2;
    let nroots = (order / 2 + 1) as u32;
    if nroots > SA01_MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

    let mut gc = run_sa01_rys_on_backend(
        backend, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, rinv, origin, exps_i,
        exps_j, coeff_i, coeff_j,
    )?;

    // s/p normalization x sa01 common_factor (0.5, intor3.c:1117).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }

    let group_out = ni_sp * nj_sp * 2;
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // REAL si transform (c2s_si_1e), NOT imaginary si_2di (Pitfall 2).
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / Sub-wave 1d): the spg-Rys/London engine class.
//
//  int1e_spgnucsp / int1e_spgsa01 COMBINE the 30-01a spgsp 8-G London structure
//  (G2E_R0I origin=ri + the rirj=ri−rj London post-multiply in the gout; NEITHER
//  reads PTR_COMMON_ORIG) with a Rys root loop (the 30-01b/c Rys+gauge engines):
//
//   - spgnucsp: NUCLEAR Rys base (atom-sum, charge −Z, int1e_type 2), g1 = D_J(g0),
//     27-product s[], 12-component London mix (intor3.c:1939-1951), rank 3, SiI.
//   - spgsa01:  RINV Rys base (single center charge +1, int1e_type 1), g1 = D_J(g0)
//     + D_I(g0) (both-side), 27-product s[], 36-component London mix
//     (intor3.c:2099-2135), rank 9, REAL Si.
//
//  Both reuse the spgsp 8-G chain helpers (sigma_p_nabla_j / x1i / x1i_of_j /
//  nabla_i_combine) + the sa01 both-side helpers, the sa01 nuclear/rinv VRR base
//  (sa01_nuc_vrr_axis), and the overlap HRR (sigma_p_hrr_axis). Headroom is
//  nmax = li+lj+3 (the D_I(R0I(D_J)) chain reaches bra li+2), one more level than
//  the sa01 kernel's li+lj+2. Rys roots are device-side; nroots is comptime and
//  the launchers fail closed (no clamp). The int1e_sp / cg / giao / spgsp paths
//  are untouched.
// ═════════════════════════════════════════════════════════════════════════════

/// Device kernel — the rank-3 spg-Rys/London `int1e_spgnucsp` assembler (NUCLEAR
/// Rys base, atom-sum). The 8-G London chain (g0..g7 with g1=D_J) feeds the
/// 12-component London `c[]·s[]` mix (intor3.c:1939-1951). `c = ri − rj` is the
/// London phase; the x1i origin is `ri` (G2E_R0I). nroots is comptime.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn spgnucsp_rys_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
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
            let nrys = nroots;
            // 8-G chain headroom: D_I(R0I(D_J)) reaches bra li+2 → nmax = li+lj+3.
            let nmax = li + lj + 3u32;
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
            let rank = 3u32;
            let total_len = rank * N_GC * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // London phase c = ri − rj (NOT common_orig).
            let cx = rix - rjx;
            let cy = riy - rjy;
            let cz = riz - rjz;

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

                    let aj2 = F::new(-2.0_f32) * aj;
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

                            sa01_nuc_vrr_axis::<F>(g, gx, c00x, rt, nmax);
                            sa01_nuc_vrr_axis::<F>(g, gy, c00y, rt, nmax);
                            sa01_nuc_vrr_axis::<F>(g, gz, c00z, rt, nmax);
                            if lj_ext >= 1u32 {
                                sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                                sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                                sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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

                                                    let bx = gx + jx * dj + ix;
                                                    let by = gy + jy * dj + iy;
                                                    let bz = gz + jz * dj + iz;

                                                    // ── 8-G chain (g1 = D_J only). ──
                                                    // X axis.
                                                    let g0xm = if ix == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        g[(bx - 1u32) as usize]
                                                    };
                                                    let g0x = g[bx as usize];
                                                    let g0xp = g[(bx + 1u32) as usize];
                                                    let g1xm = if ix == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_nabla_j::<F>(
                                                            g,
                                                            bx - 1u32,
                                                            dj,
                                                            aj2,
                                                            jx,
                                                        )
                                                    };
                                                    let g1x =
                                                        sigma_p_nabla_j::<F>(g, bx, dj, aj2, jx);
                                                    let g1xp = sigma_p_nabla_j::<F>(
                                                        g,
                                                        bx + 1u32,
                                                        dj,
                                                        aj2,
                                                        jx,
                                                    );
                                                    let g2xm = if ix == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i::<F>(g, bx - 1u32, rix)
                                                    };
                                                    let g2x = sigma_p_x1i::<F>(g, bx, rix);
                                                    let g2xp = sigma_p_x1i::<F>(g, bx + 1u32, rix);
                                                    let g3xm = if ix == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i_of_j::<F>(
                                                            g,
                                                            bx - 1u32,
                                                            dj,
                                                            aj2,
                                                            jx,
                                                            rix,
                                                        )
                                                    };
                                                    let g3x = sigma_p_x1i_of_j::<F>(
                                                        g, bx, dj, aj2, jx, rix,
                                                    );
                                                    let g3xp = sigma_p_x1i_of_j::<F>(
                                                        g,
                                                        bx + 1u32,
                                                        dj,
                                                        aj2,
                                                        jx,
                                                        rix,
                                                    );
                                                    let g4x = sigma_p_nabla_i_combine::<F>(
                                                        g0xm, g0xp, ai2, ix,
                                                    );
                                                    let g5x = sigma_p_nabla_i_combine::<F>(
                                                        g1xm, g1xp, ai2, ix,
                                                    );
                                                    let g6x = sigma_p_nabla_i_combine::<F>(
                                                        g2xm, g2xp, ai2, ix,
                                                    );
                                                    let g7x = sigma_p_nabla_i_combine::<F>(
                                                        g3xm, g3xp, ai2, ix,
                                                    );

                                                    // Y axis.
                                                    let g0ym = if iy == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        g[(by - 1u32) as usize]
                                                    };
                                                    let g0y = g[by as usize];
                                                    let g0yp = g[(by + 1u32) as usize];
                                                    let g1ym = if iy == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_nabla_j::<F>(
                                                            g,
                                                            by - 1u32,
                                                            dj,
                                                            aj2,
                                                            jy,
                                                        )
                                                    };
                                                    let g1y =
                                                        sigma_p_nabla_j::<F>(g, by, dj, aj2, jy);
                                                    let g1yp = sigma_p_nabla_j::<F>(
                                                        g,
                                                        by + 1u32,
                                                        dj,
                                                        aj2,
                                                        jy,
                                                    );
                                                    let g2ym = if iy == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i::<F>(g, by - 1u32, riy)
                                                    };
                                                    let g2y = sigma_p_x1i::<F>(g, by, riy);
                                                    let g2yp = sigma_p_x1i::<F>(g, by + 1u32, riy);
                                                    let g3ym = if iy == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i_of_j::<F>(
                                                            g,
                                                            by - 1u32,
                                                            dj,
                                                            aj2,
                                                            jy,
                                                            riy,
                                                        )
                                                    };
                                                    let g3y = sigma_p_x1i_of_j::<F>(
                                                        g, by, dj, aj2, jy, riy,
                                                    );
                                                    let g3yp = sigma_p_x1i_of_j::<F>(
                                                        g,
                                                        by + 1u32,
                                                        dj,
                                                        aj2,
                                                        jy,
                                                        riy,
                                                    );
                                                    let g4y = sigma_p_nabla_i_combine::<F>(
                                                        g0ym, g0yp, ai2, iy,
                                                    );
                                                    let g5y = sigma_p_nabla_i_combine::<F>(
                                                        g1ym, g1yp, ai2, iy,
                                                    );
                                                    let g6y = sigma_p_nabla_i_combine::<F>(
                                                        g2ym, g2yp, ai2, iy,
                                                    );
                                                    let g7y = sigma_p_nabla_i_combine::<F>(
                                                        g3ym, g3yp, ai2, iy,
                                                    );

                                                    // Z axis.
                                                    let g0zm = if iz == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        g[(bz - 1u32) as usize]
                                                    };
                                                    let g0z = g[bz as usize];
                                                    let g0zp = g[(bz + 1u32) as usize];
                                                    let g1zm = if iz == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_nabla_j::<F>(
                                                            g,
                                                            bz - 1u32,
                                                            dj,
                                                            aj2,
                                                            jz,
                                                        )
                                                    };
                                                    let g1z =
                                                        sigma_p_nabla_j::<F>(g, bz, dj, aj2, jz);
                                                    let g1zp = sigma_p_nabla_j::<F>(
                                                        g,
                                                        bz + 1u32,
                                                        dj,
                                                        aj2,
                                                        jz,
                                                    );
                                                    let g2zm = if iz == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i::<F>(g, bz - 1u32, riz)
                                                    };
                                                    let g2z = sigma_p_x1i::<F>(g, bz, riz);
                                                    let g2zp = sigma_p_x1i::<F>(g, bz + 1u32, riz);
                                                    let g3zm = if iz == 0u32 {
                                                        F::new(0.0_f32)
                                                    } else {
                                                        sigma_p_x1i_of_j::<F>(
                                                            g,
                                                            bz - 1u32,
                                                            dj,
                                                            aj2,
                                                            jz,
                                                            riz,
                                                        )
                                                    };
                                                    let g3z = sigma_p_x1i_of_j::<F>(
                                                        g, bz, dj, aj2, jz, riz,
                                                    );
                                                    let g3zp = sigma_p_x1i_of_j::<F>(
                                                        g,
                                                        bz + 1u32,
                                                        dj,
                                                        aj2,
                                                        jz,
                                                        riz,
                                                    );
                                                    let g4z = sigma_p_nabla_i_combine::<F>(
                                                        g0zm, g0zp, ai2, iz,
                                                    );
                                                    let g5z = sigma_p_nabla_i_combine::<F>(
                                                        g1zm, g1zp, ai2, iz,
                                                    );
                                                    let g6z = sigma_p_nabla_i_combine::<F>(
                                                        g2zm, g2zp, ai2, iz,
                                                    );
                                                    let g7z = sigma_p_nabla_i_combine::<F>(
                                                        g3zm, g3zp, ai2, iz,
                                                    );

                                                    // 27 cart products (intor3.c:1911-1937).
                                                    let s0 = g7x * g0y * g0z;
                                                    let s1 = g6x * g1y * g0z;
                                                    let s2 = g6x * g0y * g1z;
                                                    let s3 = g5x * g2y * g0z;
                                                    let s4 = g4x * g3y * g0z;
                                                    let s5 = g4x * g2y * g1z;
                                                    let s6 = g5x * g0y * g2z;
                                                    let s7 = g4x * g1y * g2z;
                                                    let s8 = g4x * g0y * g3z;
                                                    let s9 = g3x * g4y * g0z;
                                                    let s10 = g2x * g5y * g0z;
                                                    let s11 = g2x * g4y * g1z;
                                                    let s12 = g1x * g6y * g0z;
                                                    let s13 = g0x * g7y * g0z;
                                                    let s14 = g0x * g6y * g1z;
                                                    let s15 = g1x * g4y * g2z;
                                                    let s16 = g0x * g5y * g2z;
                                                    let s17 = g0x * g4y * g3z;
                                                    let s18 = g3x * g0y * g4z;
                                                    let s19 = g2x * g1y * g4z;
                                                    let s20 = g2x * g0y * g5z;
                                                    let s21 = g1x * g2y * g4z;
                                                    let s22 = g0x * g3y * g4z;
                                                    let s23 = g0x * g2y * g5z;
                                                    let s24 = g1x * g0y * g6z;
                                                    let s25 = g0x * g1y * g6z;
                                                    let s26 = g0x * g0y * g7z;

                                                    let elem = cj_idx * nci + ci_idx;
                                                    // 12-comp spgnucsp London mix
                                                    // (intor3.c:1953-1964) → 3 groups × gc 4.
                                                    let gp0 = base;
                                                    gc_out[(gp0 + elem) as usize] += weight
                                                        * (cy * s17 - cz * s14 - cy * s25
                                                            + cz * s22);
                                                    gc_out[(gp0 + block_len + elem) as usize] +=
                                                        weight
                                                            * (cy * s24 - cz * s21 - cy * s8
                                                                + cz * s5);
                                                    gc_out[(gp0 + 2u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cy * s7 - cz * s4 - cy * s15 + cz * s12);
                                                    gc_out[(gp0 + 3u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cy * s6 - cz * s3 + cy * s16 - cz * s13
                                                            + cy * s26
                                                            - cz * s23);

                                                    let gp1 = base + N_GC * block_len;
                                                    gc_out[(gp1 + elem) as usize] += weight
                                                        * (cz * s11 - cx * s17 - cz * s19
                                                            + cx * s25);
                                                    gc_out[(gp1 + block_len + elem) as usize] +=
                                                        weight
                                                            * (cz * s18 - cx * s24 - cz * s2
                                                                + cx * s8);
                                                    gc_out[(gp1 + 2u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cz * s1 - cx * s7 - cz * s9 + cx * s15);
                                                    gc_out[(gp1 + 3u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cz * s0 - cx * s6 + cz * s10 - cx * s16
                                                            + cz * s20
                                                            - cx * s26);

                                                    let gp2 = base + 2u32 * N_GC * block_len;
                                                    gc_out[(gp2 + elem) as usize] += weight
                                                        * (cx * s14 - cy * s11 - cx * s22
                                                            + cy * s19);
                                                    gc_out[(gp2 + block_len + elem) as usize] +=
                                                        weight
                                                            * (cx * s21 - cy * s18 - cx * s5
                                                                + cy * s2);
                                                    gc_out[(gp2 + 2u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cx * s4 - cy * s1 - cx * s12 + cy * s9);
                                                    gc_out[(gp2 + 3u32 * block_len + elem)
                                                        as usize] += weight
                                                        * (cx * s3 - cy * s0 + cx * s13 - cy * s10
                                                            + cx * s23
                                                            - cy * s20);

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

/// `g_per_axis` for one `spgnucsp_rys` class.
fn spgnucsp_rys_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj+3, lj_ext = lj+1.
    (li + lj + 4) * (lj + 2)
}

/// Evaluate every launch group of a batched `spgnucsp_rys` run (Task 35-D,
/// wave 5). One dispatch per Rys order.
fn run_spgnucsp_rys_batches<R: Runtime>(
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
    let oc_h = client.create_from_slice(f64::as_bytes(coords_src));
    let och_h = client.create_from_slice(f64::as_bytes(charges_src));
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
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    spgnucsp_rys_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(oc_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(och_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SA01_PIE4,
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

/// Backend dispatch for a whole batched `spgnucsp_rys` run.
fn dispatch_spgnucsp_rys_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_spgnucsp_rys_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_spgnucsp_rys_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_spgnucsp_rys_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_spgnucsp_rys_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_spgnucsp_rys_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
    }
}

/// One shell pair through the batched `spgnucsp_rys` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_spgnucsp_rys_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u8,
    lj: u8,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    origin_coords: &[f64],
    origin_charges: &[f64],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        (3 * N_GC) as usize,
        spgnucsp_rys_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    Ok(dispatch_spgnucsp_rys_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default())
}

/// Live `int1e_spgnucsp` Spinor launcher (rank 3, NUCLEAR Rys + 8-G London).
///
/// Builds the spg-Rys/London cart gc blocks (3 groups × 4 gc), applies the s/p
/// normalization × the spgnucsp common_factor (0.5, intor3.c:1976), folds each of
/// the 3 σ-groups through the imaginary-ket `cart_to_spinor_si_2di` (c2s_si_1ei),
/// and scatters into the contraction-major spinor grid.
///
/// The x1i origin is `ri` (G2E_R0I) and the London phase is `ri − rj`; spgnucsp
/// does NOT read PTR_COMMON_ORIG. `origin_coords`/`origin_charges` are the
/// atom-summed nuclear centers (int1e_type 2). Carries its OWN fail-closed
/// full-block (×3) staging guard + a fail-closed `nroots > MAX_DEVICE_NROOTS →
/// UnsupportedApi` guard (no clamp).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_spgnucsp_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 3;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed full-block (×3) staging guard BEFORE any write (Phase-28 CR-01,
    // T-30-01d-04). No partial guards.
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Fail-closed Rys-nroots guard (T-30-01d-05): never clamp.
    let order = li as usize + lj as usize + 2;
    let nroots = (order / 2 + 1) as u32;
    if nroots > SA01_MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

    let mut gc = run_spgnucsp_rys_on_backend(
        backend,
        nroots,
        li,
        lj,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
        ri,
        rj,
        origin_coords,
        origin_charges,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
    )?;

    // s/p normalization × spgnucsp common_factor (0.5, intor3.c:1976).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }

    let group_out = ni_sp * nj_sp * 2;
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // Imaginary-ket si transform (c2s_si_1ei) (Pitfall 2: sp/nucsp).
                cart_to_spinor_si_2di::<F>(
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Device kernel — the rank-9 spg-Rys/London `int1e_spgsa01` assembler (RINV Rys
/// base, single center charge +1). The 8-G London chain with the BOTH-SIDE g1 =
/// D_J(g0) + D_I(g0) feeds the 36-component London `c[]·s[]` mix
/// (intor3.c:2099-2135). `c = ri − rj` is the London phase; the x1i origin is `ri`
/// (G2E_R0I). nroots is comptime.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn spgsa01_rys_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
    rcx: F,
    rcy: F,
    rcz: F,
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
            let nrys = nroots;
            // spgsa01 8-G chain has a BOTH-SIDE g1 = D_J(g0) + D_I(g0); the D_I
            // term raises the bra reach, so g7 = D_I(R0I(g1)) reaches bra li+3
            // (one deeper than spgnucsp's li+2 where g1 = D_J only). Headroom
            // nmax = li+lj+4.
            let nmax = li + lj + 4u32;
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
            let rank = 9u32;
            let total_len = rank * N_GC * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                gc_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // London phase c = ri − rj (NOT common_orig).
            let cx = rix - rjx;
            let cy = riy - rjy;
            let cz = riz - rjz;

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

                    // Single rinv center (charge +1).
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

                    let fac1 = F::new(2.0_f32) * pi_const * fac / zeta;

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

                        sa01_nuc_vrr_axis::<F>(g, gx, c00x, rt, nmax);
                        sa01_nuc_vrr_axis::<F>(g, gy, c00y, rt, nmax);
                        sa01_nuc_vrr_axis::<F>(g, gz, c00z, rt, nmax);
                        if lj_ext >= 1u32 {
                            sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                            sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                            sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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

                                                let bx = gx + jx * dj + ix;
                                                let by = gy + jy * dj + iy;
                                                let bz = gz + jz * dj + iz;

                                                // ── 8-G chain (g1 = D_J + D_I both-side). ──
                                                // X axis.
                                                let g0xm = if ix == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    g[(bx - 1u32) as usize]
                                                };
                                                let g0x = g[bx as usize];
                                                let g0xp = g[(bx + 1u32) as usize];
                                                let g1xm = if ix == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_g1_bothside::<F>(
                                                        g,
                                                        bx - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        ix - 1u32,
                                                        jx,
                                                    )
                                                };
                                                let g1x = sa01_g1_bothside::<F>(
                                                    g, bx, dj, ai2, aj2, ix, jx,
                                                );
                                                let g1xp = sa01_g1_bothside::<F>(
                                                    g,
                                                    bx + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    ix + 1u32,
                                                    jx,
                                                );
                                                let g2xm = if ix == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sigma_p_x1i::<F>(g, bx - 1u32, rix)
                                                };
                                                let g2x = sigma_p_x1i::<F>(g, bx, rix);
                                                let g2xp = sigma_p_x1i::<F>(g, bx + 1u32, rix);
                                                let g3xm = if ix == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_x1i_of_g1::<F>(
                                                        g,
                                                        bx - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        ix - 1u32,
                                                        jx,
                                                        rix,
                                                    )
                                                };
                                                let g3x = sa01_x1i_of_g1::<F>(
                                                    g, bx, dj, ai2, aj2, ix, jx, rix,
                                                );
                                                let g3xp = sa01_x1i_of_g1::<F>(
                                                    g,
                                                    bx + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    ix + 1u32,
                                                    jx,
                                                    rix,
                                                );
                                                let g4x = sigma_p_nabla_i_combine::<F>(
                                                    g0xm, g0xp, ai2, ix,
                                                );
                                                let g5x = sigma_p_nabla_i_combine::<F>(
                                                    g1xm, g1xp, ai2, ix,
                                                );
                                                let g6x = sigma_p_nabla_i_combine::<F>(
                                                    g2xm, g2xp, ai2, ix,
                                                );
                                                let g7x = sigma_p_nabla_i_combine::<F>(
                                                    g3xm, g3xp, ai2, ix,
                                                );

                                                // Y axis.
                                                let g0ym = if iy == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    g[(by - 1u32) as usize]
                                                };
                                                let g0y = g[by as usize];
                                                let g0yp = g[(by + 1u32) as usize];
                                                let g1ym = if iy == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_g1_bothside::<F>(
                                                        g,
                                                        by - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        iy - 1u32,
                                                        jy,
                                                    )
                                                };
                                                let g1y = sa01_g1_bothside::<F>(
                                                    g, by, dj, ai2, aj2, iy, jy,
                                                );
                                                let g1yp = sa01_g1_bothside::<F>(
                                                    g,
                                                    by + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    iy + 1u32,
                                                    jy,
                                                );
                                                let g2ym = if iy == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sigma_p_x1i::<F>(g, by - 1u32, riy)
                                                };
                                                let g2y = sigma_p_x1i::<F>(g, by, riy);
                                                let g2yp = sigma_p_x1i::<F>(g, by + 1u32, riy);
                                                let g3ym = if iy == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_x1i_of_g1::<F>(
                                                        g,
                                                        by - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        iy - 1u32,
                                                        jy,
                                                        riy,
                                                    )
                                                };
                                                let g3y = sa01_x1i_of_g1::<F>(
                                                    g, by, dj, ai2, aj2, iy, jy, riy,
                                                );
                                                let g3yp = sa01_x1i_of_g1::<F>(
                                                    g,
                                                    by + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    iy + 1u32,
                                                    jy,
                                                    riy,
                                                );
                                                let g4y = sigma_p_nabla_i_combine::<F>(
                                                    g0ym, g0yp, ai2, iy,
                                                );
                                                let g5y = sigma_p_nabla_i_combine::<F>(
                                                    g1ym, g1yp, ai2, iy,
                                                );
                                                let g6y = sigma_p_nabla_i_combine::<F>(
                                                    g2ym, g2yp, ai2, iy,
                                                );
                                                let g7y = sigma_p_nabla_i_combine::<F>(
                                                    g3ym, g3yp, ai2, iy,
                                                );

                                                // Z axis.
                                                let g0zm = if iz == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    g[(bz - 1u32) as usize]
                                                };
                                                let g0z = g[bz as usize];
                                                let g0zp = g[(bz + 1u32) as usize];
                                                let g1zm = if iz == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_g1_bothside::<F>(
                                                        g,
                                                        bz - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        iz - 1u32,
                                                        jz,
                                                    )
                                                };
                                                let g1z = sa01_g1_bothside::<F>(
                                                    g, bz, dj, ai2, aj2, iz, jz,
                                                );
                                                let g1zp = sa01_g1_bothside::<F>(
                                                    g,
                                                    bz + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    iz + 1u32,
                                                    jz,
                                                );
                                                let g2zm = if iz == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sigma_p_x1i::<F>(g, bz - 1u32, riz)
                                                };
                                                let g2z = sigma_p_x1i::<F>(g, bz, riz);
                                                let g2zp = sigma_p_x1i::<F>(g, bz + 1u32, riz);
                                                let g3zm = if iz == 0u32 {
                                                    F::new(0.0_f32)
                                                } else {
                                                    sa01_x1i_of_g1::<F>(
                                                        g,
                                                        bz - 1u32,
                                                        dj,
                                                        ai2,
                                                        aj2,
                                                        iz - 1u32,
                                                        jz,
                                                        riz,
                                                    )
                                                };
                                                let g3z = sa01_x1i_of_g1::<F>(
                                                    g, bz, dj, ai2, aj2, iz, jz, riz,
                                                );
                                                let g3zp = sa01_x1i_of_g1::<F>(
                                                    g,
                                                    bz + 1u32,
                                                    dj,
                                                    ai2,
                                                    aj2,
                                                    iz + 1u32,
                                                    jz,
                                                    riz,
                                                );
                                                let g4z = sigma_p_nabla_i_combine::<F>(
                                                    g0zm, g0zp, ai2, iz,
                                                );
                                                let g5z = sigma_p_nabla_i_combine::<F>(
                                                    g1zm, g1zp, ai2, iz,
                                                );
                                                let g6z = sigma_p_nabla_i_combine::<F>(
                                                    g2zm, g2zp, ai2, iz,
                                                );
                                                let g7z = sigma_p_nabla_i_combine::<F>(
                                                    g3zm, g3zp, ai2, iz,
                                                );

                                                // 27 cart products (intor3.c:2071-2097).
                                                let s0 = g7x * g0y * g0z;
                                                let s1 = g6x * g1y * g0z;
                                                let s2 = g6x * g0y * g1z;
                                                let s3 = g5x * g2y * g0z;
                                                let s4 = g4x * g3y * g0z;
                                                let s5 = g4x * g2y * g1z;
                                                let s6 = g5x * g0y * g2z;
                                                let s7 = g4x * g1y * g2z;
                                                let s8 = g4x * g0y * g3z;
                                                let s9 = g3x * g4y * g0z;
                                                let s10 = g2x * g5y * g0z;
                                                let s11 = g2x * g4y * g1z;
                                                let s12 = g1x * g6y * g0z;
                                                let s13 = g0x * g7y * g0z;
                                                let s14 = g0x * g6y * g1z;
                                                let s15 = g1x * g4y * g2z;
                                                let s16 = g0x * g5y * g2z;
                                                let s17 = g0x * g4y * g3z;
                                                let s18 = g3x * g0y * g4z;
                                                let s19 = g2x * g1y * g4z;
                                                let s20 = g2x * g0y * g5z;
                                                let s21 = g1x * g2y * g4z;
                                                let s22 = g0x * g3y * g4z;
                                                let s23 = g0x * g2y * g5z;
                                                let s24 = g1x * g0y * g6z;
                                                let s25 = g0x * g1y * g6z;
                                                let s26 = g0x * g0y * g7z;

                                                let elem = cj_idx * nci + ci_idx;
                                                let nb = N_GC * block_len;
                                                // 36-comp spgsa01 London mix
                                                // (intor3.c:2100-2135) → 9 groups × gc 4.
                                                // group 0: gout 0..3
                                                gc_out[(base + 0u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cy * s16 + cz * s13 - cy * s26 + cz * s23);
                                                gc_out[(base + 0u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cy * s7 - cz * s4);
                                                gc_out[(base + 0u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cy * s8 - cz * s5);
                                                gc_out[(base + 0u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cy * s17 - cz * s14 - cy * s25 + cz * s22);
                                                // group 1: gout 4..7
                                                gc_out[(base + 1u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cy * s15 - cz * s12);
                                                gc_out[(base + 1u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cy * s26 + cz * s23 - cy * s6 + cz * s3);
                                                gc_out[(base + 1u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cy * s17 - cz * s14);
                                                gc_out[(base + 1u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cy * s8 + cz * s5 + cy * s24 - cz * s21);
                                                // group 2: gout 8..11
                                                gc_out[(base + 2u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cy * s24 - cz * s21);
                                                gc_out[(base + 2u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cy * s25 - cz * s22);
                                                gc_out[(base + 2u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cy * s6 + cz * s3 - cy * s16 + cz * s13);
                                                gc_out[(base + 2u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cy * s7 - cz * s4 - cy * s15 + cz * s12);
                                                // group 3: gout 12..15
                                                gc_out[(base + 3u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cz * s10 + cx * s16 - cz * s20 + cx * s26);
                                                gc_out[(base + 3u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cz * s1 - cx * s7);
                                                gc_out[(base + 3u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cz * s2 - cx * s8);
                                                gc_out[(base + 3u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cz * s11 - cx * s17 - cz * s19 + cx * s25);
                                                // group 4: gout 16..19
                                                gc_out[(base + 4u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cz * s9 - cx * s15);
                                                gc_out[(base + 4u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cz * s20 + cx * s26 - cz * s0 + cx * s6);
                                                gc_out[(base + 4u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cz * s11 - cx * s17);
                                                gc_out[(base + 4u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cz * s2 + cx * s8 + cz * s18 - cx * s24);
                                                // group 5: gout 20..23
                                                gc_out[(base + 5u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cz * s18 - cx * s24);
                                                gc_out[(base + 5u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cz * s19 - cx * s25);
                                                gc_out[(base + 5u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cz * s0 + cx * s6 - cz * s10 + cx * s16);
                                                gc_out[(base + 5u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cz * s1 - cx * s7 - cz * s9 + cx * s15);
                                                // group 6: gout 24..27
                                                gc_out[(base + 6u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cx * s13 + cy * s10 - cx * s23 + cy * s20);
                                                gc_out[(base + 6u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cx * s4 - cy * s1);
                                                gc_out[(base + 6u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cx * s5 - cy * s2);
                                                gc_out[(base + 6u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cx * s14 - cy * s11 - cx * s22 + cy * s19);
                                                // group 7: gout 28..31
                                                gc_out[(base + 7u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cx * s12 - cy * s9);
                                                gc_out[(base + 7u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cx * s23 + cy * s20 - cx * s3 + cy * s0);
                                                gc_out[(base + 7u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight * (cx * s14 - cy * s11);
                                                gc_out[(base + 7u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cx * s5 + cy * s2 + cx * s21 - cy * s18);
                                                // group 8: gout 32..35
                                                gc_out[(base + 8u32 * nb + 0u32 * block_len + elem)
                                                    as usize] += weight * (cx * s21 - cy * s18);
                                                gc_out[(base + 8u32 * nb + 1u32 * block_len + elem)
                                                    as usize] += weight * (cx * s22 - cy * s19);
                                                gc_out[(base + 8u32 * nb + 2u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (-cx * s3 + cy * s0 - cx * s13 + cy * s10);
                                                gc_out[(base + 8u32 * nb + 3u32 * block_len + elem)
                                                    as usize] += weight
                                                    * (cx * s4 - cy * s1 - cx * s12 + cy * s9);

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

/// `g_per_axis` for one `spgsa01_rys` class.
fn spgsa01_rys_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj+4, lj_ext = lj+1.
    (li + lj + 5) * (lj + 2)
}

/// Evaluate every launch group of a batched `spgsa01_rys` run (Task 35-D,
/// wave 5). One dispatch per Rys order; the rinv centre is a per-call scalar.
fn run_spgsa01_rys_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rinv: [f64; 3],
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
                    spgsa01_rys_kernel::launch_unchecked::<f64, R>(
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
                        rinv[0],
                        rinv[1],
                        rinv[2],
                        SA01_PIE4,
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

/// Backend dispatch for a whole batched `spgsa01_rys` run.
fn dispatch_spgsa01_rys_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rinv: [f64; 3],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_spgsa01_rys_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, rinv)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_spgsa01_rys_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rinv)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_spgsa01_rys_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, rinv)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_spgsa01_rys_batches::<cubecl_hip::HipRuntime>(client, basis, groups, rinv)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_spgsa01_rys_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rinv)
        }
    }
}

/// One shell pair through the batched `spgsa01_rys` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_spgsa01_rys_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u8,
    lj: u8,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rinv: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
        li as u32,
        lj as u32,
        nprim_i as u32,
        nprim_j as u32,
        nctr_i as u32,
        nctr_j as u32,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        (9 * N_GC) as usize,
        spgsa01_rys_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    Ok(
        dispatch_spgsa01_rys_batches(backend, &handles, std::slice::from_ref(&group), rinv)
            .pop()
            .unwrap_or_default(),
    )
}

/// Live `int1e_spgsa01` Spinor launcher (rank 9, RINV Rys + 8-G London).
///
/// Builds the rank-9 spg-Rys/London cart gc blocks (9 groups × 4 gc), applies the
/// s/p normalization × the spgsa01 common_factor (0.5, intor3.c:2184), folds each
/// of the 9 σ-groups through the REAL `cart_to_spinor_si_2d` (c2s_si_1e — NOT the
/// imaginary si_2di), and scatters into the contraction-major spinor grid.
///
/// The x1i origin is `ri` (G2E_R0I) and the London phase is `ri − rj`; spgsa01
/// does NOT read PTR_COMMON_ORIG. `rinv` is the single rinv center
/// (env[PTR_RINV_ORIG], charge +1, int1e_type 1). Carries its OWN fail-closed
/// full-block (×9) staging guard + a fail-closed `nroots > MAX_DEVICE_NROOTS →
/// UnsupportedApi` guard (no clamp).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_spgsa01_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rinv: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    const RANK: usize = 9;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = RANK * (N_GC as usize) * block_len;

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed full-block (×9) staging guard BEFORE any write (Phase-28 CR-01,
    // T-30-01d-04). No partial guards.
    let staging_required = ni_sp * nj_sp * 2 * RANK;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Fail-closed Rys-nroots guard (T-30-01d-05): never clamp.
    let order = li as usize + lj as usize + 4;
    let nroots = (order / 2 + 1) as u32;
    if nroots > SA01_MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

    let mut gc = run_spgsa01_rys_on_backend(
        backend, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, rinv, exps_i, exps_j,
        coeff_i, coeff_j,
    )?;

    // s/p normalization × spgsa01 common_factor (0.5, intor3.c:2184).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj) * 0.5;
    for v in gc.iter_mut() {
        *v *= sp_scale;
    }

    let group_out = ni_sp * nj_sp * 2;
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..RANK {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                // REAL si transform (c2s_si_1e), NOT imaginary si_2di (Pitfall 2).
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

                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Device-vs-host parity for a single primitive p-shell pair (non-square p×d).
    ///
    /// CpuRuntime FP-env side effect (Pitfall 5) can perturb host f64 ~1e-11; the
    /// kernel itself is bit-faithful so the band is set just above that, well
    /// below any real numerical divergence.
    #[test]
    fn sigma_p_device_matches_host() {
        let cases: &[(u32, u32, f64, f64)] = &[
            (1, 2, 1.3, 0.7), // p × d  (non-square — surfaces orientation bugs)
            (1, 1, 0.9, 1.1), // p × p
            (0, 0, 0.8, 0.6), // s × s (sanity: g1 = -2*ai*g[1])
            (2, 1, 1.5, 0.4), // d × p
        ];
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.7];
        let coeff_i = [0.9_f64];
        let coeff_j = [1.1_f64];

        for &(li, lj, ai, aj) in cases {
            let exps_i_c = [ai];
            let exps_j_c = [aj];

            let host = sigma_p_host(
                1, li, lj, &exps_i_c, &exps_j_c, &coeff_i, &coeff_j, 1, 1, ri, rj,
            );
            let dev = run_sigma_p_on_backend(
                &ResolvedBackend::Cpu(cpu_client()),
                1,
                li,
                lj,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &exps_i_c,
                &exps_j_c,
                &coeff_i,
                &coeff_j,
            )
            .expect("sigma_p batch");

            assert_eq!(host.len(), dev.len(), "length mismatch li={li} lj={lj}");
            for (k, (h, d)) in host.iter().zip(dev.iter()).enumerate() {
                assert!(
                    (h - d).abs() <= 1e-9 * (1.0 + h.abs()),
                    "sigma_p device-vs-host mismatch at li={li} lj={lj} idx={k}: host={h} dev={d}"
                );
            }
        }
    }

    /// s × s sanity: nf=1, four gc blocks; assert the scalar slot (gc_1, block 3)
    /// is identically 0.0 for every cart n (int1e_sp scalar slot is ZERO).
    #[test]
    fn sigma_p_device_matches_host_ss_scalar_slot_zero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.9];
        let exps_i = [0.85_f64];
        let exps_j = [0.55_f64];
        let coeff_i = [1.0_f64];
        let coeff_j = [1.0_f64];

        let li = 0u32;
        let lj = 0u32;
        let nci = 1usize;
        let ncj = 1usize;
        let block_len = nci * ncj;

        let dev = run_sigma_p_on_backend(
            &ResolvedBackend::Cpu(cpu_client()),
            1,
            li,
            lj,
            1,
            1,
            1,
            1,
            ri,
            rj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
        )
        .expect("sigma_p batch");

        // 4 gc blocks each of length block_len (=1).
        assert_eq!(dev.len(), 4 * block_len);
        // Scalar slot is block index 3.
        for n in 0..block_len {
            let scalar = dev[3 * block_len + n];
            assert_eq!(scalar, 0.0, "gc_1 scalar slot must be 0.0 at n={n}");
        }
        // And the Pauli blocks are non-trivial (g1 = -2*ai*g[1] ≠ 0 in general).
        let any_nonzero = (0..3 * block_len).any(|i| dev[i].abs() > 0.0);
        assert!(
            any_nonzero,
            "Pauli gc_x/gc_y/gc_z blocks should be non-zero"
        );
    }

    /// Layout: block boundaries land at `comp*(nf*ictr*jctr)` (component-blocked),
    /// NOT interleaved `n*4+comp`. Verify with nctr=1 p×d (block_len = 3*6 = 18):
    /// the 4 gc blocks are contiguous, scalar block last and all-zero.
    #[test]
    fn sigma_p_layout_is_component_blocked() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.2];
        let exps_i = [1.1_f64];
        let exps_j = [0.7_f64];
        let coeff_i = [1.0_f64];
        let coeff_j = [1.0_f64];

        let li = 1u32; // p → nci=3
        let lj = 2u32; // d → ncj=6
        let nci = 3usize;
        let ncj = 6usize;
        let block_len = nci * ncj; // 18

        let dev = run_sigma_p_on_backend(
            &ResolvedBackend::Cpu(cpu_client()),
            1,
            li,
            lj,
            1,
            1,
            1,
            1,
            ri,
            rj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
        )
        .expect("sigma_p batch");

        // 4 contiguous component-leading blocks (component-blocked, not n*4+comp).
        assert_eq!(dev.len(), 4 * block_len);
        // Scalar block (block 3) is entirely zero.
        for n in 0..block_len {
            assert_eq!(
                dev[3 * block_len + n],
                0.0,
                "scalar gc_1 block must be all-zero at n={n}"
            );
        }
        // Cross-check against the host reference (same layout).
        let host = sigma_p_host(
            1, li, lj, &exps_i, &exps_j, &coeff_i, &coeff_j, 1, 1, ri, rj,
        );
        for (k, (h, d)) in host.iter().zip(dev.iter()).enumerate() {
            assert!(
                (h - d).abs() <= 1e-9 * (1.0 + h.abs()),
                "layout test mismatch at idx={k}: host={h} dev={d}"
            );
        }
    }
}
