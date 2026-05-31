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

use crate::backend::ResolvedBackend;
use crate::transform::c2spinor::{cart_to_spinor_si_2d, spinor_len};
use crate::transform::c2s::ncart;
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
//  Device kernel — `#[cube(launch)]` — generic σ·p G-tensor assembler.
//
//  Single UNIT_POS==0 work item; in-kernel pdata recompute in F (norm_i =
//  norm_j = 1.0); zeroed scratch + out; (pi,pj) primitive loop × (ci,cj)
//  contraction loop — cloning `one_electron_grad_bra_kernel`'s structure.
//
//  G-tensor sizing: nmax = li+lj+1, lj_ext = lj (one extra bra level for the
//  ix+1 nabla read). The output is `N_GC`-block COMPONENT-LEADING per (ci,cj)
//  block — see module docs.
// ─────────────────────────────────────────────────────────────────────────────

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn sigma_p_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    gc_out: &mut Array<F>,
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
    #[comptime] tensor_rank: u32,
) {
    if UNIT_POS == 0u32 {
        // G-tensor sizing (overlap base, one extra bra level for the nabla read).
        let nmax = li + lj + 1u32;
        let lj_ext = lj;
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        // n_gc = N_GC gc blocks per tensor component.
        let n_blocks = tensor_rank * N_GC;
        let total_len = n_blocks * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        // Zero the full output buffer (the scalar gc_1 block is written 0 too).
        let mut oi = 0u32;
        while oi < out_total {
            gc_out[oi as usize] = F::new(0.0);
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

                sigma_p_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                sigma_p_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                sigma_p_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj_ext >= 1u32 {
                    sigma_p_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    sigma_p_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    sigma_p_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
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
                                        gc_out[b1 as usize] += F::new(0.0);

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

/// Dispatch [`sigma_p_kernel`] at `f64` on a resolved backend's client.
///
/// Returns the `N_GC`-block component-leading gc accumulator of length
/// `nctr_i * nctr_j * tensor_rank * N_GC * nci * ncj`. Buffer creation mirrors
/// [`one_electron.rs::run_1e_grad_bra_device`]; `tensor_rank` selects the
/// monomorphization at the `launch::<f64, R>` call site (comptime).
#[allow(clippy::too_many_arguments)]
fn run_sigma_p_device<R: Runtime>(
    client: &ComputeClient<R>,
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
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 1;
    let lj_ext_u = lj_u;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let n_blocks = (tensor_rank as usize) * (N_GC as usize);
    let out_len = (nctr_i as usize) * (nctr_j as usize) * n_blocks * nci * ncj;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    // Scratch + output buffers.
    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($rank:expr) => {
            sigma_p_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), total_g) },
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
                $rank,
            )
        };
    }

    // Only tensor_rank==1 (int1e_sp) is exercised this phase; rank 3 (int1e_sigma)
    // lands in Phase 29 and routes through the same comptime path.
    if tensor_rank == 1 {
        launch_with!(1u32);
    } else {
        launch_with!(3u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_sigma_p_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
///
/// Mirrors `one_electron.rs::run_1e_grad_bra_on_backend`. Driven live by
/// [`launch_int1e_sp_spinor_pair`] (the FND-05 path), which feeds these gc
/// blocks into `cart_to_spinor_si_2d`.
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
    let out = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_sigma_p_device::<cubecl::cpu::CpuRuntime>(
            client,
            tensor_rank,
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
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_sigma_p_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            tensor_rank,
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
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_sigma_p_device::<cubecl_cuda::CudaRuntime>(
            client,
            tensor_rank,
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
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_sigma_p_device::<cubecl_hip::HipRuntime>(
            client,
            tensor_rank,
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
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_sigma_p_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            tensor_rank,
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
        ),
    };
    Ok(out)
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
            let dev = run_sigma_p_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client(),
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
            );

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

        let dev = run_sigma_p_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
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
        );

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

        let dev = run_sigma_p_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
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
        );

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
