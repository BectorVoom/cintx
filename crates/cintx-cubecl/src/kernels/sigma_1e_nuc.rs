//! Phase 29 Wave 1 — nuclear-engine 1e Group-4 σ families (spnucsp/srnucsr/sprinvsp).
//!
//! These three families route through the 2-electron Rys nuclear G-tensor
//! (`G2E_D_I/D_J` for spnucsp/sprinvsp, `G2E_R_I/R_J` for srnucsr) with the
//! rank-4 σ·p G-tensor gout `(s5-s7, s6-s2, s1-s3, s0+s4+s8)`. They are wired
//! through the real bra-σ-mix transform `cart_to_spinor_si_2d` (c2s_si_1e).
//!
//!   spnucsp  — type-2 nuclear attraction (atom-sum, charge −Z), G2E_D
//!   srnucsr  — type-2 nuclear attraction, G2E_R
//!   sprinvsp — type-1 rinv (single center, charge +1), G2E_D

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
use crate::kernels::one_electron::{
    ONE_E_DERIV_SHAPE_STRIDE, OneEDerivLaunchGroup, one_e_deriv_single_pair_group,
    one_e_g_slab_stride, one_e_launch_geometry, one_e_per_unit,
};
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// PIE4 = pi/4 (Rys weight normalization, matches `one_electron::PIE4`).
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const PIE4: f64 = 0.78539816339744827900_f64;
const N_GC: u32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
//  Per-axis 2e VRR/HRR helpers (root-dependent c00/b10), cloned from
//  one_electron's nuclear path.
// ─────────────────────────────────────────────────────────────────────────────

#[cube]
fn nuc_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, c00: F, b10: F, nmax: u32) {
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

#[cube]
fn nuc_hrr_axis<F: Float>(g: &mut Array<F>, base: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
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

#[cube]
fn nuc_nabla_j<F: Float>(g: &Array<F>, idx0: u32, dj: u32, aj2: F, jexp: u32) -> F {
    if jexp == 0u32 {
        aj2 * g[(idx0 + dj) as usize]
    } else {
        F::cast_from(jexp) * g[(idx0 - dj) as usize] + aj2 * g[(idx0 + dj) as usize]
    }
}

#[cube]
fn nuc_nabla_i<F: Float>(g: &Array<F>, idx0: u32, ai2: F, iexp: u32) -> F {
    if iexp == 0u32 {
        ai2 * g[(idx0 + 1u32) as usize]
    } else {
        F::cast_from(iexp) * g[(idx0 - 1u32) as usize] + ai2 * g[(idx0 + 1u32) as usize]
    }
}

#[cube]
fn nuc_nabla_ij<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    ai2: F,
    aj2: F,
    iexp: u32,
    jexp: u32,
) -> F {
    if iexp == 0u32 {
        let g1_ip = nuc_nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
        ai2 * g1_ip
    } else {
        let g1_im = nuc_nabla_j::<F>(g, idx0 - 1u32, dj, aj2, jexp);
        let g1_ip = nuc_nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
        F::cast_from(iexp) * g1_im + ai2 * g1_ip
    }
}

/// `*_J` operator at `idx0`: R-tensor index shift (`g[idx0+dj]`) when `use_r==1`,
/// else D-tensor j-nabla.
#[cube]
fn op_j<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    aj2: F,
    jexp: u32,
    #[comptime] use_r: u32,
) -> F {
    if comptime!(use_r == 1u32) {
        g[(idx0 + dj) as usize]
    } else {
        nuc_nabla_j::<F>(g, idx0, dj, aj2, jexp)
    }
}

/// `*_I` operator at `idx0`: R-tensor index shift (`g[idx0+1]`) when `use_r==1`,
/// else D-tensor i-nabla.
#[cube]
fn op_i<F: Float>(g: &Array<F>, idx0: u32, ai2: F, iexp: u32, #[comptime] use_r: u32) -> F {
    if comptime!(use_r == 1u32) {
        g[(idx0 + 1u32) as usize]
    } else {
        nuc_nabla_i::<F>(g, idx0, ai2, iexp)
    }
}

/// `*_I(*_J)` composed operator at `idx0`: R-tensor double shift (`g[idx0+dj+1]`)
/// when `use_r==1`, else D-tensor composed nabla.
#[cube]
#[allow(clippy::too_many_arguments)]
fn op_ij<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    ai2: F,
    aj2: F,
    iexp: u32,
    jexp: u32,
    #[comptime] use_r: u32,
) -> F {
    if comptime!(use_r == 1u32) {
        g[(idx0 + dj + 1u32) as usize]
    } else {
        nuc_nabla_ij::<F>(g, idx0, dj, ai2, aj2, iexp, jexp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 30 (GIAO-03 / Sub-wave 1b): gauge `x1i`-with-origin device helpers on the
//  Rys nuclear G-tensor. These mirror the overlap-engine `sigma_p::sigma_p_x1i`
//  recurrence `f[i] = g[i+1] + origin·g[i]` (CINTx1i_2e, g2e.c:4779; G2E_RCI macro
//  g2e.h:98), but operate on this module's nuclear g-tensor layout (bra raise =
//  `+1`, ket nabla = `±dj`). `origin = dri = ri − common_orig` (cg) or `[0,0,0]`
//  (giao, where x1i collapses to G2E_R_I = a pure +1 bra shift).
// ─────────────────────────────────────────────────────────────────────────────

/// `G2E_RCI` gauge `x1i`-with-origin at index `idx0`: `g[idx0+1] + origin·g[idx0]`.
/// `origin = 0` reduces this to `g[idx0+1]` = `G2E_R_I` (the giao natural center).
#[cube]
fn nuc_x1i<F: Float>(g: &Array<F>, idx0: u32, origin: F) -> F {
    g[(idx0 + 1u32) as usize] + origin * g[idx0 as usize]
}

/// `G2E_RCI` of the ket-nabla `g1 = G2E_D_J(g0)`: apply `∇_j` first at the two
/// bra-stencil points `idx0`/`idx0+1`, then the `x1i`-with-origin combine.
#[cube]
fn nuc_x1i_of_j<F: Float>(g: &Array<F>, idx0: u32, dj: u32, aj2: F, jexp: u32, origin: F) -> F {
    let g1_i = nuc_nabla_j::<F>(g, idx0, dj, aj2, jexp);
    let g1_ip = nuc_nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
    g1_ip + origin * g1_i
}

// ─────────────────────────────────────────────────────────────────────────────
//  Nuclear σ kernel. family: 0 = D-tensor (spnucsp/sprinvsp), 1 = R-tensor
//  (srnucsr). gout rank 4 → 1 σ-group (N_GC blocks). nmax = li+lj+2,
//  lj_ext = lj+1 (composed +1/+1 headroom). Origins parameterized.
// ─────────────────────────────────────────────────────────────────────────────

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_nuc_kernel<F: Float + CubeElement>(
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
    #[comptime] use_r: u32,
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
            let nmax = li + lj + 2u32;
            let lj_ext = lj + 1u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = N_GC * block_len;
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

                        #[unroll]
                        for irys in 0..nroots {
                            let u_n = urys[irys as usize];
                            let w_n = wrys[irys as usize];
                            let tau = u_n / (F::new(1.0_f32) + u_n);
                            let rt = aij2 * (F::new(1.0_f32) - tau);

                            let c00x = (px - rix) + tau * crijx;
                            let c00y = (py - riy) + tau * crijy;
                            let c00z = (pz - riz) + tau * crijz;

                            let mut gi = gbase;
                            while gi < gbase + 3u32 * (nmax + 1u32) * (lj_ext + 1u32) {
                                g[gi as usize] = F::new(0.0_f32);
                                gi += 1u32;
                            }
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
                            g[gz as usize] = fac1 * w_n;

                            nuc_vrr_axis::<F>(g, gx, c00x, rt, nmax);
                            nuc_vrr_axis::<F>(g, gy, c00y, rt, nmax);
                            nuc_vrr_axis::<F>(g, gz, c00z, rt, nmax);
                            if lj_ext >= 1u32 {
                                nuc_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                                nuc_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                                nuc_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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

                                                    // g1=*_J, g2=*_I, g3=*_I(*_J), per axis
                                                    // (R-tensor index shifts vs D-tensor nabla,
                                                    // selected by comptime use_r).
                                                    let g1x = op_j::<F>(g, bx, dj, aj2, jx, use_r);
                                                    let g2x = op_i::<F>(g, bx, ai2, ix, use_r);
                                                    let g3x = op_ij::<F>(
                                                        g, bx, dj, ai2, aj2, ix, jx, use_r,
                                                    );
                                                    let g1y = op_j::<F>(g, by, dj, aj2, jy, use_r);
                                                    let g2y = op_i::<F>(g, by, ai2, iy, use_r);
                                                    let g3y = op_ij::<F>(
                                                        g, by, dj, ai2, aj2, iy, jy, use_r,
                                                    );
                                                    let g1z = op_j::<F>(g, bz, dj, aj2, jz, use_r);
                                                    let g2z = op_i::<F>(g, bz, ai2, iz, use_r);
                                                    let g3z = op_ij::<F>(
                                                        g, bz, dj, ai2, aj2, iz, jz, use_r,
                                                    );

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
                                                    gc_out[(base + elem) as usize] +=
                                                        weight * (s5 - s7);
                                                    gc_out[(base + block_len + elem) as usize] +=
                                                        weight * (s6 - s2);
                                                    gc_out[(base + 2u32 * block_len + elem)
                                                        as usize] += weight * (s1 - s3);
                                                    gc_out[(base + 3u32 * block_len + elem)
                                                        as usize] += weight * (s0 + s4 + s8);

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

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / Sub-wave 1b): Rys+gauge nuclear σ kernel
//  (cg_sa10nucsp / giao_sa10nucsp). Same Rys nuclear G-tensor as sigma_nuc_kernel
//  (G2E_* int1e_type 2, nuclear attraction), but the per-root gout applies the
//  gauge `x1i`-with-origin fold INSIDE the Rys root loop:
//
//      g0 = nuclear base, g1 = G2E_D_J(g0)  (ket nabla, intor3.c:1242)
//      g2 = G2E_RCI/R_I(g0, origin)         (gauge x1i, intor3.c:1243)
//      g3 = G2E_RCI/R_I(g1, origin)         (gauge x1i of ket-nabla, intor3.c:1244)
//
//  9 products s[0..8] (transcribed VERBATIM from intor3.c:1252-1260) fold into the
//  12-component (3 tensor × gc 4-block) cg/giao nucsp gout (intor3.c:1263-1274) —
//  the cg and giao gout BODIES are byte-identical; only `origin` differs
//  (dri = ri−common_orig for cg, [0,0,0] for giao). rank 3, c2s_si_1ei (imaginary).
//  Headroom identical to sigma_nuc_kernel (nmax = li+lj+2, lj_ext = lj+1).
// ═════════════════════════════════════════════════════════════════════════════

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_nuc_gauge_kernel<F: Float + CubeElement>(
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
    gauge_x: F,
    gauge_y: F,
    gauge_z: F,
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
            // rank 3 → 3 σ-groups (component-leading), each N_GC blocks.
            let total_len = 3u32 * N_GC * block_len;
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

                            let mut gi = gbase;
                            while gi < gbase + total_g {
                                g[gi as usize] = F::new(0.0_f32);
                                gi += 1u32;
                            }
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
                            g[gz as usize] = fac1 * w_n;

                            nuc_vrr_axis::<F>(g, gx, c00x, rt, nmax);
                            nuc_vrr_axis::<F>(g, gy, c00y, rt, nmax);
                            nuc_vrr_axis::<F>(g, gz, c00z, rt, nmax);
                            if lj_ext >= 1u32 {
                                nuc_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                                nuc_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                                nuc_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                            }

                            // Accumulate this root's 12-comp cg/giao nucsp gout into
                            // every (ci,cj). The gauge x1i-with-origin fold is applied
                            // HERE, inside the Rys root loop.
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
                                                    // g1 = G2E_D_J(g0) (ket nabla).
                                                    let g1x = nuc_nabla_j::<F>(g, bx, dj, aj2, jx);
                                                    let g1y = nuc_nabla_j::<F>(g, by, dj, aj2, jy);
                                                    let g1z = nuc_nabla_j::<F>(g, bz, dj, aj2, jz);
                                                    // g2 = G2E_RCI/R_I(g0, origin) (gauge x1i).
                                                    let g2x = nuc_x1i::<F>(g, bx, gauge_x);
                                                    let g2y = nuc_x1i::<F>(g, by, gauge_y);
                                                    let g2z = nuc_x1i::<F>(g, bz, gauge_z);
                                                    // g3 = G2E_RCI/R_I(g1, origin) (gauge x1i of ∇_j).
                                                    let g3x = nuc_x1i_of_j::<F>(
                                                        g, bx, dj, aj2, jx, gauge_x,
                                                    );
                                                    let g3y = nuc_x1i_of_j::<F>(
                                                        g, by, dj, aj2, jy, gauge_y,
                                                    );
                                                    let g3z = nuc_x1i_of_j::<F>(
                                                        g, bz, dj, aj2, jz, gauge_z,
                                                    );

                                                    // 9 cart products (intor3.c:1252-1260).
                                                    let s0 = g3x * g0y * g0z;
                                                    let s1 = g2x * g1y * g0z;
                                                    let s2 = g2x * g0y * g1z;
                                                    let s3 = g1x * g2y * g0z;
                                                    let s4 = g0x * g3y * g0z;
                                                    let s5 = g0x * g2y * g1z;
                                                    let s6 = g1x * g0y * g2z;
                                                    let s7 = g0x * g1y * g2z;
                                                    let s8 = g0x * g0y * g3z;

                                                    // 12-comp cg/giao nucsp gout
                                                    // (intor3.c:1263-1274), byte-identical
                                                    // to the overlap cg gout mix.
                                                    let elem = cj_idx * nci + ci_idx;
                                                    let gg0 = base;
                                                    let gg1 = base + N_GC * block_len;
                                                    let gg2 = base + 2u32 * N_GC * block_len;
                                                    gc_out[(gg0 + elem) as usize] +=
                                                        weight * (s8 + s4);
                                                    gc_out[(gg0 + block_len + elem) as usize] +=
                                                        weight * (-s3);
                                                    gc_out[(gg0 + 2u32 * block_len + elem)
                                                        as usize] += weight * (-s6);
                                                    gc_out[(gg0 + 3u32 * block_len + elem)
                                                        as usize] += weight * (s7 - s5);
                                                    gc_out[(gg1 + elem) as usize] += weight * (-s1);
                                                    gc_out[(gg1 + block_len + elem) as usize] +=
                                                        weight * (s0 + s8);
                                                    gc_out[(gg1 + 2u32 * block_len + elem)
                                                        as usize] += weight * (-s7);
                                                    gc_out[(gg1 + 3u32 * block_len + elem)
                                                        as usize] += weight * (s2 - s6);
                                                    gc_out[(gg2 + elem) as usize] += weight * (-s2);
                                                    gc_out[(gg2 + block_len + elem) as usize] +=
                                                        weight * (-s5);
                                                    gc_out[(gg2 + 2u32 * block_len + elem)
                                                        as usize] += weight * (s4 + s0);
                                                    gc_out[(gg2 + 3u32 * block_len + elem)
                                                        as usize] += weight * (s3 - s1);

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

/// `g_per_axis` for one σ·nuclear class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn sigma_nuc_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj+2, lj_ext = lj+1. Shared by the plain and gauge kernels.
    (li + lj + 3) * (lj + 2)
}

/// Evaluate every launch group of a batched gauge σ·nuclear run (Task 35-D,
/// wave 5). One dispatch per Rys order; the gauge origin is a per-call scalar.
#[allow(clippy::too_many_arguments)]
fn run_sigma_nuc_gauge_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    gauge: [f64; 3],
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
            one_e_launch_geometry::<R>(n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>());

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    sigma_nuc_gauge_kernel::launch_unchecked::<f64, R>(
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
                        PIE4,
                        std::f64::consts::PI,
                        gauge[0],
                        gauge[1],
                        gauge[2],
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

/// Backend dispatch for a whole batched gauge σ·nuclear run.
#[allow(clippy::too_many_arguments)]
fn dispatch_sigma_nuc_gauge_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    gauge: [f64; 3],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_sigma_nuc_gauge_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            gauge,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_sigma_nuc_gauge_batches::<cubecl_wgpu::WgpuRuntime>(
                client,
                basis,
                groups,
                gauge,
                origin_coords,
                origin_charges,
            )
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_sigma_nuc_gauge_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            gauge,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_sigma_nuc_gauge_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            gauge,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_sigma_nuc_gauge_batches::<cubecl_wgpu::WgpuRuntime>(
                client,
                basis,
                groups,
                gauge,
                origin_coords,
                origin_charges,
            )
        }
    }
}

/// Build Rys+gauge nuclear σ cart gc blocks for `cg_sa10nucsp` / `giao_sa10nucsp`
/// (rank 3, 12-comp gout). `gauge` is `dri = ri − common_orig` (cg) or `[0,0,0]`
/// (giao). `origin_coords`/`origin_charges` are the nuclear attraction centers
/// (charge factor −Z per atom), precomputed by the caller. Fail-closed
/// `nroots > MAX_DEVICE_NROOTS` (no Rys-nroots clamp), mirroring
/// `run_sigma_nuc_on_backend`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sigma_nuc_gauge_on_backend(
    backend: &ResolvedBackend,
    li: u8,
    lj: u8,
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
) -> Result<Vec<f64>, cintxRsError> {
    // Fail-closed nroots guard — never clamp (CR-01 lesson). The composed +1/+1
    // double-derivative order is li+lj+2; nroots = order/2 + 1.
    const MAX_DEVICE_NROOTS: u32 = 5;
    let order = li as usize + lj as usize + 2;
    let nroots = (order / 2 + 1) as u32;
    if nroots > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

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
        sigma_nuc_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    Ok(dispatch_sigma_nuc_gauge_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        gauge,
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default())
}

/// Evaluate every launch group of a batched σ·nuclear run (Task 35-D, wave 5).
///
/// Two comptime parameters: `nroots` is the merge key, and `use_r` is fixed by
/// the caller's operator (`srnucsr` vs the rest).
fn run_sigma_nuc_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    use_r: u32,
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
            one_e_launch_geometry::<R>(n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>());

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr, $ur:expr) => {
                unsafe {
                    sigma_nuc_kernel::launch_unchecked::<f64, R>(
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
                        PIE4,
                        std::f64::consts::PI,
                        norig,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        $ur,
                        per_unit,
                    );
                }
            };
        }

        match (group.nroots, use_r) {
            (1, 0) => launch_with!(1u32, 0u32),
            (2, 0) => launch_with!(2u32, 0u32),
            (3, 0) => launch_with!(3u32, 0u32),
            (4, 0) => launch_with!(4u32, 0u32),
            (_, 0) => launch_with!(5u32, 0u32),
            (1, _) => launch_with!(1u32, 1u32),
            (2, _) => launch_with!(2u32, 1u32),
            (3, _) => launch_with!(3u32, 1u32),
            (4, _) => launch_with!(4u32, 1u32),
            (_, _) => launch_with!(5u32, 1u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched σ·nuclear run.
fn dispatch_sigma_nuc_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    use_r: u32,
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_sigma_nuc_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            use_r,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_sigma_nuc_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            use_r,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_sigma_nuc_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            use_r,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_sigma_nuc_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            use_r,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_sigma_nuc_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            use_r,
            origin_coords,
            origin_charges,
        ),
    }
}

/// Build nuclear-engine σ cart gc blocks for `op` (spnucsp/srnucsr/sprinvsp).
///
/// `origin_coords` (norig*3, x/y/z per origin) and `origin_charges` (norig, the
/// charge factor: −Z for spnucsp/srnucsr atom-sum, +1 for sprinvsp single
/// center) are precomputed by the caller (the cubecl crate does not depend on
/// cintx-compat for the raw atm/bas/env slot constants).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sigma_nuc_on_backend(
    op: &str,
    backend: &ResolvedBackend,
    li: u8,
    lj: u8,
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
) -> Result<Vec<f64>, cintxRsError> {
    // nroots = (li+lj+2)/2 + 1 (standard 1e nuclear rule for the composed +1/+1
    // double-derivative order li+lj+2). The kernel only emits comptime root
    // counts 1..=5, so fail closed for higher l rather than silently truncating
    // — matching the sibling kernels' fail-closed contract.
    const MAX_DEVICE_NROOTS: u32 = 5;
    let order = li as usize + lj as usize + 2;
    let nroots = (order / 2 + 1) as u32;
    if nroots > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

    let use_r = if op == "srnucsr" { 1u32 } else { 0u32 };
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
        N_GC as usize,
        sigma_nuc_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    Ok(dispatch_sigma_nuc_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        use_r,
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default())
}
