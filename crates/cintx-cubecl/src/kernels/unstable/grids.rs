//! `grids` family: grid-point integrals with the NGRIDS env parameter
//! (`cint1e_grids.c`).
//!
//! HOST/DEVICE SPLIT (quick-260529-twi, Phase 21 D-04 honest split):
//!
//! ON DEVICE (`grids_scalar_kernel<F>`, generic over `F: Float`): the SCALAR
//! nuclear-like `int1e_grids_sph` per-(grid-point, primitive-pair) Cartesian
//! G-tensor build — the Rys-quadrature VRR (`vrr_2e`) + bra→ket HRR + the
//! `gx*gy*gz` Cartesian contraction (the body of `grids_contract_nuclear_like`),
//! in f64-internal / F-output. The pair data (center_p, zeta_ab, aij2, fac) is
//! computed host-side via `compute_pdata_host` and passed as scalars; the kernel
//! is dispatched once per (grid, prim-pair) and the host accumulates over
//! contractions, exactly mirroring `launch_grids_kernel`'s loop.
//!
//! ON HOST (host part of the split, UNCHANGED): the `cart_to_sph_1e` transform +
//! the grids-layout AO scatter into `staging`, plus the `compute_pdata_host`
//! pair-data setup.
//!
//! ON DEVICE (DERIVATIVES, quick-260530-9ay): the derivative variants
//! `grids_ip` / `_ipip` / `_ipvip` / `_spvsp` now run their per-(grid, prim-pair)
//! Cartesian block on the device via the single parameterized
//! `grids_deriv_kernel<F>` (`#[comptime] op_kind` selects ip/ipip/ipvip/spvsp).
//! It builds the Rys-accumulated 3-axis G-tensor, applies the nabla_i / nabla_j /
//! σ ladders (`grids_nabla_i_axis` / `grids_nabla_j_axis`, device ports of
//! `nabla_i_host`/`nabla_j_host`), and emits the comp-major Cartesian block
//! matching the host `grids_contract_*` layout (including the ipip
//! column-transposed component order and the spvsp 9→4 combine). The host keeps
//! the accumulate + `cart_to_sph_1e` + grids-layout scatter UNCHANGED. Pairs that
//! need `nroots>2` fall back to the host reference (device wires `rys_root{1,2}`).
//!
//! EVAL_RAW WIRING (fixed in fix/general-contraction-nctr-1e): the public
//! `eval_raw(RawApiId::Symbol("int1e_grids_sph"))` path now accepts the grids
//! 4-element shell tuple `[i, j, grid_start, grid_end]` (libcint reads only
//! shls[0..2] and loops the full env grid set; the window entries are ignored)
//! and folds NGRIDS into the output sizing, so the grids family is driven
//! end-to-end through `eval_raw` like every other family. The CPU vendor-parity
//! gate is `unstable_source_parity.rs::grids_parity`; the ROCm vendor-parity
//! oracle is `tests/grids_random_rocm_parity.rs`. An in-crate device-vs-host
//! parity (`run_grids_nuclear_device::<HipRuntime>` vs the host
//! `grids_contract_nuclear_like`) remains as a backend-only cross-check.
//!
//! RYS-ROOT ORDERING (fixed alongside the wiring): the derivative kernels
//! (`grids_deriv_kernel` + the host `grids_contract_ip/_ipip/_ipvip/_spvsp`)
//! apply the nabla ladders and form the gx·gy·gz products PER Rys root and
//! accumulate the PRODUCT across roots — matching the scalar path and libcint.
//! The earlier formulation summed the per-axis G-tensor over roots first, which
//! computes (ΣGx)·(ΣGy)·(ΣGz) ≠ Σ(Gx·Gy·Gz) for nroots>1 (every s/p case with
//! a second Rys root, e.g. the derivative operators on p shells).

use super::shared::{apply_nabla_i_3axis, apply_nabla_j_3axis, cart_comps, common_fac_sp};
use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_2e_step_host};
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::{rys_root1, rys_root2, rys_roots_host};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats, planner::GridsEnvParams};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Rys `PIE4 = pi/4` passed into the device `rys_root{1,2}` kernels.
const GRIDS_PIE4: f64 = 0.78539816339744827900_f64;
/// Fail-closed device guard: the scalar grids path uses nroots ∈ {1,2}.
const GRIDS_MAX_DEVICE_NROOTS: u32 = 2;

/// Compute the Rys-quadrature G-tensor for one primitive pair and one "nuclear" center.
///
/// Uses the same algorithm as nuclear attraction in one_electron.rs, but with
/// a user-supplied center `rc` (grid point) instead of atomic coordinates.
/// The prefactor is `2*pi * fac / zeta` (no atomic charge factor).
///
/// Returns accumulated Cartesian integral buffer of size nci * ncj.
fn grids_contract_nuclear_like(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    li: u8,
    lj: u8,
    nmax_extra: u32, // extra j-levels for derivative operators (0 for base)
    lj_extra: u32,   // extra HRR ket levels for derivative operators
) -> Vec<f64> {
    let nmax = (li + lj) as u32 + nmax_extra;
    let lj_hrr = lj as u32 + lj_extra;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let mut out = vec![0.0_f64; nci * ncj];

    let nrys_roots = (li + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj_hrr + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];

    // Vector from P to grid center: crij = rc - rp
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];

    // Boys argument x = zeta * |P - rc|^2
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    // Get Rys roots and weights
    // Rys roots/weights sized to nrys_roots (1..=5). The device kernels wire only
    // rys_root{1,2} (GRIDS_MAX_DEVICE_NROOTS); this host fallback handles nroots>2,
    // so the arrays MUST have nrys_roots entries (not a fixed 2) or the loop below
    // indexes out of bounds. rys_roots_host is byte-identical to rys_root{1,2}_host
    // for nroots<=2 and correct for 3..=5 (fail-closed >5 — deferred Wheeler).
    let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

    // Grids prefactor: 2*pi * fac / zeta (same as nuclear but no -Z_c charge factor)
    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;

    // For each Rys root
    for n in 0..nrys_roots as usize {
        let u_n = u_arr[n];
        let w_n = w_arr[n];

        let tau = u_n / (1.0 + u_n);

        // Modified recurrence coefficient b10 = aij2 * (1 - tau)
        let rt = pd.aij2 * (1.0 - tau);

        // VRR c00: (P - ri) + tau * crij = (P - ri) + tau * (rc - rp)
        let c00 = [
            (rp[0] - ri[0]) + tau * crij[0],
            (rp[1] - ri[1]) + tau * crij[1],
            (rp[2] - ri[2]) + tau * crij[2],
        ];

        let gz0_root = fac1 * w_n;

        let mut g_root = vec![0.0_f64; 3 * g_per_axis];

        let gx_off = 0;
        let gy_off = g_per_axis;
        let gz_off = 2 * g_per_axis;

        g_root[gx_off] = 1.0;
        g_root[gy_off] = 1.0;
        g_root[gz_off] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(
                &mut g_root[gx_off..gx_off + g_per_axis],
                c00[0],
                rt,
                nmax,
                1,
            );
            vrr_2e_step_host(
                &mut g_root[gy_off..gy_off + g_per_axis],
                c00[1],
                rt,
                nmax,
                1,
            );
            vrr_2e_step_host(
                &mut g_root[gz_off..gz_off + g_per_axis],
                c00[2],
                rt,
                nmax,
                1,
            );
        }

        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        if lj_hrr >= 1 {
            let di = 1u32;
            let dj_stride = nmax + 1;
            hrr_step_host(
                &mut g_root[gx_off..gx_off + g_per_axis],
                rirj[0],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
            hrr_step_host(
                &mut g_root[gy_off..gy_off + g_per_axis],
                rirj[1],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
            hrr_step_host(
                &mut g_root[gz_off..gz_off + g_per_axis],
                rirj[2],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
        }

        // Contract this root's contribution (base operator: gx*gy*gz)
        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let vx = g_root[gx_off + jx as usize * dj + ix as usize];
                let vy = g_root[gy_off + jy as usize * dj + iy as usize];
                let vz = g_root[gz_off + jz as usize * dj + iz as usize];
                out[ci_idx * ncj + cj_idx] += vx * vy * vz;
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  DEVICE: grids scalar nuclear-like kernel (#[cube(launch)], generic over F).
//  On-device port of `grids_contract_nuclear_like` for ONE primitive pair and
//  ONE grid point — Rys VRR + bra→ket HRR + gx*gy*gz Cartesian contraction.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-axis 2e vertical recurrence (stride 1): mirrors `vrr_2e_step_host`.
#[cube]
fn grids_vrr_axis<F: Float>(g: &mut Array<F>, off: u32, c00: F, b10: F, nmax: u32) {
    g[(off + 1u32) as usize] = c00 * g[off as usize];
    let mut n = 1u32;
    while n < nmax {
        g[(off + n + 1u32) as usize] =
            F::cast_from(n) * b10 * g[(off + n - 1u32) as usize] + c00 * g[(off + n) as usize];
        n += 1u32;
    }
}

/// Per-axis horizontal recurrence (di=1, ket stride `dj`): mirrors `hrr_step_host`.
#[cube]
fn grids_hrr_axis<F: Float>(g: &mut Array<F>, off: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
    let mut j = 1u32;
    while j <= lj {
        let i_max = li_max - j;
        let mut i = 0u32;
        while i <= i_max {
            let idx_out = off + j * dj + i;
            let idx_hi = off + (j - 1u32) * dj + (i + 1u32);
            let idx_lo = off + (j - 1u32) * dj + i;
            g[idx_out as usize] = g[idx_hi as usize] + rirj * g[idx_lo as usize];
            i += 1u32;
        }
        j += 1u32;
    }
}

/// On-device `grids_contract_nuclear_like` for ONE primitive pair + ONE grid
/// point. Writes the `nci*ncj` Cartesian block (i-major: `out[ci*ncj + cj]`),
/// accumulated over the Rys roots. Pair data (rp=center_p, zeta_ab, aij2, fac)
/// is host-computed (via `compute_pdata_host`) and passed as scalars.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grids_scalar_kernel<F: Float + CubeElement>(
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
    rcx: F,
    rcy: F,
    rcz: F,
    rpx: F,
    rpy: F,
    rpz: F,
    zeta_ab: F,
    aij2: F,
    fac: F,
    pie4: F,
    two_pi: F,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let block = nci * ncj;
        let mut oi = 0u32;
        while oi < block {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let gx_off = 0u32;
        let gy_off = g_per_axis;
        let gz_off = 2u32 * g_per_axis;

        let crijx = rcx - rpx;
        let crijy = rcy - rpy;
        let crijz = rcz - rpz;
        let x_boys = zeta_ab * (crijx * crijx + crijy * crijy + crijz * crijz);

        if comptime!(nroots == 1u32) {
            rys_root1::<F>(x_boys, urys, wrys, pie4);
        } else {
            rys_root2::<F>(x_boys, urys, wrys, pie4);
        }

        let fac1 = two_pi * fac / zeta_ab;
        let total_g = 3u32 * g_per_axis;

        let nrys = nroots;
        let mut nr: u32 = 0u32;
        while nr < nrys {
            let mut gi = 0u32;
            while gi < total_g {
                g[gi as usize] = F::new(0.0);
                gi += 1u32;
            }

            let u_n = urys[nr as usize];
            let w_n = wrys[nr as usize];
            let tau = u_n / (F::new(1.0) + u_n);
            let rt = aij2 * (F::new(1.0) - tau);

            let c00x = (rpx - rix) + tau * crijx;
            let c00y = (rpy - riy) + tau * crijy;
            let c00z = (rpz - riz) + tau * crijz;

            g[gx_off as usize] = F::new(1.0);
            g[gy_off as usize] = F::new(1.0);
            g[gz_off as usize] = fac1 * w_n;

            if nmax >= 1u32 {
                grids_vrr_axis::<F>(g, gx_off, c00x, rt, nmax);
                grids_vrr_axis::<F>(g, gy_off, c00y, rt, nmax);
                grids_vrr_axis::<F>(g, gz_off, c00z, rt, nmax);
            }

            if lj_hrr >= 1u32 {
                grids_hrr_axis::<F>(g, gx_off, rix - rjx, dj, nmax, lj_hrr);
                grids_hrr_axis::<F>(g, gy_off, riy - rjy, dj, nmax, lj_hrr);
                grids_hrr_axis::<F>(g, gz_off, riz - rjz, dj, nmax, lj_hrr);
            }

            // Cartesian contraction — ci outer, cj inner, lx descending (matches
            // host `cart_comps` + `out[ci_idx*ncj + cj_idx]`).
            let mut ci_idx = 0u32;
            let mut ia = 0u32;
            while ia <= li {
                let ix = li - ia;
                let li_minus_ix = li - ix;
                let mut ib = 0u32;
                while ib <= li_minus_ix {
                    let iy = li_minus_ix - ib;
                    let iz = li - ix - iy;

                    let mut cj_idx = 0u32;
                    let mut ja = 0u32;
                    while ja <= lj {
                        let jx = lj - ja;
                        let lj_minus_jx = lj - jx;
                        let mut jb = 0u32;
                        while jb <= lj_minus_jx {
                            let jy = lj_minus_jx - jb;
                            let jz = lj - jx - jy;

                            let vx = g[(gx_off + jx * dj + ix) as usize];
                            let vy = g[(gy_off + jy * dj + iy) as usize];
                            let vz = g[(gz_off + jz * dj + iz) as usize];
                            cart_out[(ci_idx * ncj + cj_idx) as usize] += vx * vy * vz;

                            cj_idx += 1u32;
                            jb += 1u32;
                        }
                        ja += 1u32;
                    }
                    ci_idx += 1u32;
                    ib += 1u32;
                }
                ia += 1u32;
            }

            nr += 1u32;
        }
    }
}

/// Host dispatcher: run `grids_scalar_kernel` for ONE prim-pair + ONE grid point
/// on runtime `R`, returning the `nci*ncj` f64 Cartesian block.
#[allow(clippy::too_many_arguments)]
fn run_grids_nuclear_device<R: Runtime>(
    client: &ComputeClient<R>,
    nroots: u32,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rp: [f64; 3],
    zeta_ab: f64,
    aij2: f64,
    fac: f64,
) -> Vec<f64> {
    let out_len = (nci * ncj) as usize;
    let g_zero = vec![0.0_f64; 3 * g_per_axis as usize];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots as usize];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));
    let two_pi = 2.0 * std::f64::consts::PI;

    macro_rules! launch_with {
        ($nr:expr) => {
            grids_scalar_kernel::launch::<f64, R>(
                client,
                crate::plane::single_cube_count(),
                crate::plane::backend_plane_cube_dim::<R>(),
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_per_axis as usize) },
                unsafe { ArrayArg::from_raw_parts(u_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(w_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                rc[0],
                rc[1],
                rc[2],
                rp[0],
                rp[1],
                rp[2],
                zeta_ab,
                aij2,
                fac,
                GRIDS_PIE4,
                two_pi,
                li,
                lj,
                nmax,
                lj_hrr,
                dj,
                g_per_axis,
                nci,
                ncj,
                $nr,
            )
        };
    }
    if nroots == 1 {
        launch_with!(1u32);
    } else {
        launch_with!(2u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for the grids scalar nuclear-like device kernel
/// (Cpu / Wgpu / Cuda / ROCm-HIP / Metal). Rocm → `cubecl_hip::HipRuntime`.
#[allow(clippy::too_many_arguments)]
fn run_grids_nuclear_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rp: [f64; 3],
    zeta_ab: f64,
    aij2: f64,
    fac: f64,
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_grids_nuclear_device::<cubecl::cpu::CpuRuntime>(
            client, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj, rc, rp,
            zeta_ab, aij2, fac,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_grids_nuclear_device::<cubecl_wgpu::WgpuRuntime>(
            client, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj, rc, rp,
            zeta_ab, aij2, fac,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_grids_nuclear_device::<cubecl_cuda::CudaRuntime>(
            client, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj, rc, rp,
            zeta_ab, aij2, fac,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_grids_nuclear_device::<cubecl_hip::HipRuntime>(
            client, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj, rc, rp,
            zeta_ab, aij2, fac,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_grids_nuclear_device::<cubecl_wgpu::WgpuRuntime>(
            client, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj, rc, rp,
            zeta_ab, aij2, fac,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DEVICE: grids DERIVATIVE kernels (#[cube(launch)], generic over F).
//  On-device port of `grids_contract_ip / _ipip / _ipvip / _spvsp` for ONE
//  primitive pair and ONE grid point. ONE parameterized kernel selects the op
//  via `#[comptime] op_kind` (1=ip, 2=ipip, 3=ipvip, 4=spvsp). Mirrors the host
//  derivative fns exactly: build the Rys-accumulated 3-axis G-tensor (g0), apply
//  the nabla / σ ladders into scratch axes (g1/g2/g3), then the comp-major
//  Cartesian contraction. f64-internal, F-output. Pair data (rp, zeta_ab, aij2,
//  fac) is host-computed via `compute_pdata_host` and passed as scalars; the
//  kernel runs once per (grid, prim-pair) and the host accumulates contractions,
//  mirroring `launch_grids_kernel`.
//
//  VENDOR ORACLE: now driven through `eval_raw` (the grids 4-element shell-tuple
//  wiring is fixed) — see `unstable_source_parity.rs::grids_parity` (CPU) and
//  `tests/grids_random_rocm_parity.rs` (ROCm). A DIRECT device-vs-host parity
//  (the `run_grids_<op>_device::<R>` dispatchers vs the host
//  `grids_contract_<op>`) is also kept in-crate (CpuRuntime) and on
//  `cubecl_hip::HipRuntime` as a backend-only cross-check.
// ─────────────────────────────────────────────────────────────────────────────

/// op_kind discriminants for `grids_deriv_kernel`.
const GRIDS_OP_IP: u32 = 1;
const GRIDS_OP_IPIP: u32 = 2;
const GRIDS_OP_IPVIP: u32 = 3;
const GRIDS_OP_SPVSP: u32 = 4;

/// Per-axis nabla_i (bra gradient): mirrors `nabla_i_host`.
/// `D_i[j,i] = i*g[j,i-1] - 2*ai*g[j,i+1]`, with `i=0 → -2*ai*g[j,1]`.
/// Writes `df[off..]`, reads `g[off..]`; both use ket-stride `dj`. Source must
/// carry `li+1` bra levels. `lj` here is the ket range to fill.
#[cube]
fn grids_nabla_i_axis<F: Float>(
    df: &mut Array<F>,
    g: &Array<F>,
    off: u32,
    ai: F,
    li: u32,
    lj: u32,
    dj: u32,
) {
    let ai2 = F::new(-2.0) * ai;
    let lj_max = lj;
    let li_max = li;
    let mut j = 0u32;
    while j <= lj_max {
        // i = 0
        df[(off + j * dj) as usize] = ai2 * g[(off + j * dj + 1u32) as usize];
        let mut i = 1u32;
        while i <= li_max {
            let val = F::cast_from(i) * g[(off + j * dj + (i - 1u32)) as usize]
                + ai2 * g[(off + j * dj + (i + 1u32)) as usize];
            df[(off + j * dj + i) as usize] = val;
            i += 1u32;
        }
        j += 1u32;
    }
}

/// Per-axis nabla_j (ket gradient): mirrors `nabla_j_host`.
/// `D_j[j,i] = j*g[j-1,i] - 2*aj*g[j+1,i]`, with `j=0 → -2*aj*g[1,i]`.
/// Source must carry `lj+1` ket levels.
#[cube]
fn grids_nabla_j_axis<F: Float>(
    df: &mut Array<F>,
    g: &Array<F>,
    off: u32,
    aj: F,
    li: u32,
    lj: u32,
    dj: u32,
) {
    let aj2 = F::new(-2.0) * aj;
    let li_max = li;
    let lj_max = lj;
    // j = 0
    let mut i = 0u32;
    while i <= li_max {
        df[(off + i) as usize] = aj2 * g[(off + dj + i) as usize];
        i += 1u32;
    }
    let mut j = 1u32;
    while j <= lj_max {
        let mut i2 = 0u32;
        while i2 <= li_max {
            let val = F::cast_from(j) * g[(off + (j - 1u32) * dj + i2) as usize]
                + aj2 * g[(off + (j + 1u32) * dj + i2) as usize];
            df[(off + j * dj + i2) as usize] = val;
            i2 += 1u32;
        }
        j += 1u32;
    }
}

/// On-device grids derivative kernel for ONE prim-pair + ONE grid point.
///
/// `op_kind` (comptime): 1=ip (3 comp), 2=ipip (9), 3=ipvip (9), 4=spvsp (4).
/// Writes `cart_out` with `ncomp*nci*ncj` elements, comp-major
/// (`cart_out[comp*nci*ncj + ci*ncj + cj]`), MATCHING the host `grids_contract_*`
/// layout (including the ipip column-transposed component order).
///
/// Scratch axes `g0/g1/g2/g3` are host-allocated `3*g_per_axis` buffers (passed
/// as args, not device-local — same idiom as the scalar kernel). `g0` is the
/// Rys-accumulated base tensor; `g1/g2/g3` hold the nabla ladders per op.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grids_deriv_kernel<F: Float + CubeElement>(
    g0: &mut Array<F>,
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
    rcx: F,
    rcy: F,
    rcz: F,
    rpx: F,
    rpy: F,
    rpz: F,
    zeta_ab: F,
    aij2: F,
    fac: F,
    ai: F,
    aj: F,
    pie4: F,
    two_pi: F,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    #[comptime] nroots: u32,
    #[comptime] op_kind: u32,
) {
    if UNIT_POS == 0u32 {
        let gx_off = 0u32;
        let gy_off = g_per_axis;
        let gz_off = 2u32 * g_per_axis;
        let total_g = 3u32 * g_per_axis;

        // Zero g0 accumulator + cart_out.
        let mut zi = 0u32;
        while zi < total_g {
            g0[zi as usize] = F::new(0.0);
            zi += 1u32;
        }
        let mut ncomp = 9u32;
        if comptime!(op_kind == GRIDS_OP_IP) {
            ncomp = 3u32;
        }
        if comptime!(op_kind == GRIDS_OP_SPVSP) {
            ncomp = 4u32;
        }
        let block = ncomp * nci * ncj;
        let mut oi = 0u32;
        while oi < block {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let crijx = rcx - rpx;
        let crijy = rcy - rpy;
        let crijz = rcz - rpz;
        let x_boys = zeta_ab * (crijx * crijx + crijy * crijy + crijz * crijz);

        if comptime!(nroots == 1u32) {
            rys_root1::<F>(x_boys, urys, wrys, pie4);
        } else {
            rys_root2::<F>(x_boys, urys, wrys, pie4);
        }

        let fac1 = two_pi * fac / zeta_ab;
        let nrys = nroots;

        // Rys quadrature of the derivative: the nabla ladders (g1/g2/g3) and the
        // gx·gy·gz products must be formed PER ROOT and the product accumulated
        // across roots. Summing the per-axis G-tensor over roots first and then
        // forming the product computes (ΣGx)·(ΣGy)·(ΣGz) ≠ Σ(Gx·Gy·Gz) for
        // nroots>1 (the scalar kernel already accumulates per-root products).
        let nij = nci * ncj;
        let mut nr: u32 = 0u32;
        while nr < nrys {
            // Build THIS root's 3-axis G-tensor into g0.
            let mut gi = 0u32;
            while gi < total_g {
                g0[gi as usize] = F::new(0.0);
                gi += 1u32;
            }

            let u_n = urys[nr as usize];
            let w_n = wrys[nr as usize];
            let tau = u_n / (F::new(1.0) + u_n);
            let rt = aij2 * (F::new(1.0) - tau);

            let c00x = (rpx - rix) + tau * crijx;
            let c00y = (rpy - riy) + tau * crijy;
            let c00z = (rpz - riz) + tau * crijz;

            g0[gx_off as usize] = F::new(1.0);
            g0[gy_off as usize] = F::new(1.0);
            g0[gz_off as usize] = fac1 * w_n;

            if nmax >= 1u32 {
                grids_vrr_axis::<F>(g0, gx_off, c00x, rt, nmax);
                grids_vrr_axis::<F>(g0, gy_off, c00y, rt, nmax);
                grids_vrr_axis::<F>(g0, gz_off, c00z, rt, nmax);
            }

            if lj_hrr >= 1u32 {
                grids_hrr_axis::<F>(g0, gx_off, rix - rjx, dj, nmax, lj_hrr);
                grids_hrr_axis::<F>(g0, gy_off, riy - rjy, dj, nmax, lj_hrr);
                grids_hrr_axis::<F>(g0, gz_off, riz - rjz, dj, nmax, lj_hrr);
            }

            // ── Apply the per-op nabla / σ ladders into g1/g2/g3 (this root) ──
            // Zero scratch axes before reuse.
            let mut z1 = 0u32;
            while z1 < total_g {
                g1[z1 as usize] = F::new(0.0);
                g2[z1 as usize] = F::new(0.0);
                g3[z1 as usize] = F::new(0.0);
                z1 += 1u32;
            }

            if comptime!(op_kind == GRIDS_OP_IP) {
                // g1 = D_i(g0), bra li, ket lj.
                grids_nabla_i_axis::<F>(g1, g0, gx_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g1, g0, gy_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g1, g0, gz_off, ai, li, lj, dj);
            } else if comptime!(op_kind == GRIDS_OP_IPIP) {
                // g1 = D_i(g0) [bra li+1], g2 = D_i(g0) [bra li], g3 = D_i(g1) [bra li].
                grids_nabla_i_axis::<F>(g1, g0, gx_off, ai, li + 1u32, lj, dj);
                grids_nabla_i_axis::<F>(g1, g0, gy_off, ai, li + 1u32, lj, dj);
                grids_nabla_i_axis::<F>(g1, g0, gz_off, ai, li + 1u32, lj, dj);
                grids_nabla_i_axis::<F>(g2, g0, gx_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g2, g0, gy_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g2, g0, gz_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g3, g1, gx_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g3, g1, gy_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g3, g1, gz_off, ai, li, lj, dj);
            } else {
                // ipvip + spvsp share the G-tensor.
                // g1 = D_j(g0) [bra li+1, ket lj], g2 = D_i(g0) [bra li, ket lj+1],
                // g3 = D_i(g1) [bra li, ket lj].
                grids_nabla_j_axis::<F>(g1, g0, gx_off, aj, li + 1u32, lj, dj);
                grids_nabla_j_axis::<F>(g1, g0, gy_off, aj, li + 1u32, lj, dj);
                grids_nabla_j_axis::<F>(g1, g0, gz_off, aj, li + 1u32, lj, dj);
                grids_nabla_i_axis::<F>(g2, g0, gx_off, ai, li, lj + 1u32, dj);
                grids_nabla_i_axis::<F>(g2, g0, gy_off, ai, li, lj + 1u32, dj);
                grids_nabla_i_axis::<F>(g2, g0, gz_off, ai, li, lj + 1u32, dj);
                grids_nabla_i_axis::<F>(g3, g1, gx_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g3, g1, gy_off, ai, li, lj, dj);
                grids_nabla_i_axis::<F>(g3, g1, gz_off, ai, li, lj, dj);
            }

            // ── Cartesian contraction (ci outer, cj inner; lx descending) ─────
            let mut ci_idx = 0u32;
            let mut ia = 0u32;
            while ia <= li {
                let ix = li - ia;
                let li_minus_ix = li - ix;
                let mut ib = 0u32;
                while ib <= li_minus_ix {
                    let iy = li_minus_ix - ib;
                    let iz = li - ix - iy;

                    let mut cj_idx = 0u32;
                    let mut ja = 0u32;
                    while ja <= lj {
                        let jx = lj - ja;
                        let lj_minus_jx = lj - jx;
                        let mut jb = 0u32;
                        while jb <= lj_minus_jx {
                            let jy = lj_minus_jx - jb;
                            let jz = lj - jx - jy;

                            let g0x = g0[(gx_off + jx * dj + ix) as usize];
                            let g0y = g0[(gy_off + jy * dj + iy) as usize];
                            let g0z = g0[(gz_off + jz * dj + iz) as usize];
                            let g1x = g1[(gx_off + jx * dj + ix) as usize];
                            let g1y = g1[(gy_off + jy * dj + iy) as usize];
                            let g1z = g1[(gz_off + jz * dj + iz) as usize];

                            let base = ci_idx * ncj + cj_idx;

                            if comptime!(op_kind == GRIDS_OP_IP) {
                                cart_out[(0u32 * nij + base) as usize] += g1x * g0y * g0z;
                                cart_out[(1u32 * nij + base) as usize] += g0x * g1y * g0z;
                                cart_out[(2u32 * nij + base) as usize] += g0x * g0y * g1z;
                            } else {
                                let g2x = g2[(gx_off + jx * dj + ix) as usize];
                                let g2y = g2[(gy_off + jy * dj + iy) as usize];
                                let g2z = g2[(gz_off + jz * dj + iz) as usize];
                                let g3x = g3[(gx_off + jx * dj + ix) as usize];
                                let g3y = g3[(gy_off + jy * dj + iy) as usize];
                                let g3z = g3[(gz_off + jz * dj + iz) as usize];

                                let s0 = g3x * g0y * g0z;
                                let s1 = g2x * g1y * g0z;
                                let s2 = g2x * g0y * g1z;
                                let s3 = g1x * g2y * g0z;
                                let s4 = g0x * g3y * g0z;
                                let s5 = g0x * g2y * g1z;
                                let s6 = g1x * g0y * g2z;
                                let s7 = g0x * g1y * g2z;
                                let s8 = g0x * g0y * g3z;

                                if comptime!(op_kind == GRIDS_OP_IPIP) {
                                    // libcint column-transposed: s0 s3 s6 s1 s4 s7 s2 s5 s8
                                    cart_out[(0u32 * nij + base) as usize] += s0;
                                    cart_out[(1u32 * nij + base) as usize] += s3;
                                    cart_out[(2u32 * nij + base) as usize] += s6;
                                    cart_out[(3u32 * nij + base) as usize] += s1;
                                    cart_out[(4u32 * nij + base) as usize] += s4;
                                    cart_out[(5u32 * nij + base) as usize] += s7;
                                    cart_out[(6u32 * nij + base) as usize] += s2;
                                    cart_out[(7u32 * nij + base) as usize] += s5;
                                    cart_out[(8u32 * nij + base) as usize] += s8;
                                } else if comptime!(op_kind == GRIDS_OP_IPVIP) {
                                    cart_out[(0u32 * nij + base) as usize] += s0;
                                    cart_out[(1u32 * nij + base) as usize] += s1;
                                    cart_out[(2u32 * nij + base) as usize] += s2;
                                    cart_out[(3u32 * nij + base) as usize] += s3;
                                    cart_out[(4u32 * nij + base) as usize] += s4;
                                    cart_out[(5u32 * nij + base) as usize] += s5;
                                    cart_out[(6u32 * nij + base) as usize] += s6;
                                    cart_out[(7u32 * nij + base) as usize] += s7;
                                    cart_out[(8u32 * nij + base) as usize] += s8;
                                } else {
                                    // spvsp: gout = [s5-s7, s6-s2, s1-s3, s0+s4+s8].
                                    // The combine is linear in the per-root products
                                    // s0..s8, so accumulating (s5-s7) etc. per root is
                                    // identical to combining the root-summed products.
                                    cart_out[(0u32 * nij + base) as usize] += s5 - s7;
                                    cart_out[(1u32 * nij + base) as usize] += s6 - s2;
                                    cart_out[(2u32 * nij + base) as usize] += s1 - s3;
                                    cart_out[(3u32 * nij + base) as usize] += s0 + s4 + s8;
                                }
                            }

                            cj_idx += 1u32;
                            jb += 1u32;
                        }
                        ja += 1u32;
                    }
                    ci_idx += 1u32;
                    ib += 1u32;
                }
                ia += 1u32;
            }

            nr += 1u32;
        }
    }
}

/// Host dispatcher: run `grids_deriv_kernel` for ONE prim-pair + ONE grid point
/// on runtime `R`, returning the `ncomp*nci*ncj` f64 Cartesian block.
#[allow(clippy::too_many_arguments)]
fn run_grids_deriv_device<R: Runtime>(
    client: &ComputeClient<R>,
    op_kind: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    ncomp: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rp: [f64; 3],
    zeta_ab: f64,
    aij2: f64,
    fac: f64,
    ai: f64,
    aj: f64,
) -> Vec<f64> {
    let out_len = (ncomp * nci * ncj) as usize;
    let axis_len = 3 * g_per_axis as usize;
    let g_zero = vec![0.0_f64; axis_len];
    let g0_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g2_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g3_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots as usize];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));
    let two_pi = 2.0 * std::f64::consts::PI;

    macro_rules! launch_with {
        ($nr:expr, $op:expr) => {
            grids_deriv_kernel::launch::<f64, R>(
                client,
                crate::plane::single_cube_count(),
                crate::plane::backend_plane_cube_dim::<R>(),
                unsafe { ArrayArg::from_raw_parts(g0_h.clone(), axis_len) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), axis_len) },
                unsafe { ArrayArg::from_raw_parts(g2_h.clone(), axis_len) },
                unsafe { ArrayArg::from_raw_parts(g3_h.clone(), axis_len) },
                unsafe { ArrayArg::from_raw_parts(u_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(w_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                rc[0],
                rc[1],
                rc[2],
                rp[0],
                rp[1],
                rp[2],
                zeta_ab,
                aij2,
                fac,
                ai,
                aj,
                GRIDS_PIE4,
                two_pi,
                li,
                lj,
                nmax,
                lj_hrr,
                dj,
                g_per_axis,
                nci,
                ncj,
                $nr,
                $op,
            )
        };
    }
    // nroots and op_kind are comptime; enumerate the (nroots, op) matrix.
    macro_rules! dispatch_op {
        ($nr:expr) => {
            match op_kind {
                GRIDS_OP_IP => launch_with!($nr, GRIDS_OP_IP),
                GRIDS_OP_IPIP => launch_with!($nr, GRIDS_OP_IPIP),
                GRIDS_OP_IPVIP => launch_with!($nr, GRIDS_OP_IPVIP),
                _ => launch_with!($nr, GRIDS_OP_SPVSP),
            }
        };
    }
    if nroots == 1 {
        dispatch_op!(1u32);
    } else {
        dispatch_op!(2u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for the grids derivative device kernel
/// (Cpu / Wgpu / Cuda / ROCm-HIP / Metal). Rocm → `cubecl_hip::HipRuntime`.
#[allow(clippy::too_many_arguments)]
fn run_grids_deriv_on_backend(
    backend: &ResolvedBackend,
    op_kind: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    nmax: u32,
    lj_hrr: u32,
    dj: u32,
    g_per_axis: u32,
    nci: u32,
    ncj: u32,
    ncomp: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rp: [f64; 3],
    zeta_ab: f64,
    aij2: f64,
    fac: f64,
    ai: f64,
    aj: f64,
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_grids_deriv_device::<cubecl::cpu::CpuRuntime>(
            client, op_kind, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp, ri, rj,
            rc, rp, zeta_ab, aij2, fac, ai, aj,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_grids_deriv_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_kind, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp, ri, rj,
            rc, rp, zeta_ab, aij2, fac, ai, aj,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_grids_deriv_device::<cubecl_cuda::CudaRuntime>(
            client, op_kind, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp, ri, rj,
            rc, rp, zeta_ab, aij2, fac, ai, aj,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_grids_deriv_device::<cubecl_hip::HipRuntime>(
            client, op_kind, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp, ri, rj,
            rc, rp, zeta_ab, aij2, fac, ai, aj,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_grids_deriv_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_kind, nroots, li, lj, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp, ri, rj,
            rc, rp, zeta_ab, aij2, fac, ai, aj,
        ),
    }
}

/// Compute grids-ip (nabla_i) Cartesian integral for one primitive pair at one grid point.
///
/// Returns 3 * nci * ncj elements (3 Cartesian derivative components: x, y, z).
/// Output layout: [comp][ci][cj] where comp=0..2 is the nabla_i direction.
fn grids_contract_ip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IP requires G-tensor built with li+1 bra levels to allow nabla_i.
    // nmax for VRR = (li+1) + lj (we add 1 to bra for derivative)
    let nmax = (li + 1 + lj) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 1 + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    // Rys roots/weights sized to nrys_roots (1..=5). The device kernels wire only
    // rys_root{1,2} (GRIDS_MAX_DEVICE_NROOTS); this host fallback handles nroots>2,
    // so the arrays MUST have nrys_roots entries (not a fixed 2) or the loop below
    // indexes out of bounds. rys_roots_host is byte-identical to rys_root{1,2}_host
    // for nroots<=2 and correct for 3..=5 (fail-closed >5 — deferred Wheeler).
    let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // Rys quadrature of the gradient: I = Σ_root [ ∂x(Gx)·Gy·Gz + Gx·∂y(Gy)·Gz +
    // Gx·Gy·∂z(Gz) ]. The nabla ladder and the gx·gy·gz product MUST be applied
    // per root and the PRODUCT accumulated across roots. Summing the per-axis
    // G-tensor over roots first and then forming the product computes
    // (ΣGx)·(ΣGy)·(ΣGz), which differs from Σ(Gx·Gy·Gz) whenever nrys_roots > 1.
    let mut out = vec![0.0_f64; 3 * nci * ncj];

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

        let mut g0 = vec![0.0_f64; 3 * g_per_axis];
        g0[0] = 1.0;
        g0[g_per_axis] = 1.0;
        g0[2 * g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g0[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[g_per_axis..2 * g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[2 * g_per_axis..3 * g_per_axis], c00[2], rt, nmax, 1);
        }

        if lj >= 1 {
            hrr_step_host(
                &mut g0[0..g_per_axis],
                rirj[0],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
            hrr_step_host(
                &mut g0[g_per_axis..2 * g_per_axis],
                rirj[1],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
            hrr_step_host(
                &mut g0[2 * g_per_axis..3 * g_per_axis],
                rirj[2],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
        }

        // Apply nabla_i to THIS root's G-tensor.
        let g1 = apply_nabla_i_3axis(&g0, ai, li as u32, lj as u32, nmax, g_per_axis);

        // Contract: 3 components (x, y, z), accumulating the product per root.
        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let g0x = g0[jx as usize * dj + ix as usize];
                let g0y = g0[g_per_axis + jy as usize * dj + iy as usize];
                let g0z = g0[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g1x = g1[jx as usize * dj + ix as usize];
                let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
                let g1z = g1[2 * g_per_axis + jz as usize * dj + iz as usize];

                let base_idx = ci_idx * ncj + cj_idx;
                out[0 * nci * ncj + base_idx] += g1x * g0y * g0z; // comp x
                out[1 * nci * ncj + base_idx] += g0x * g1y * g0z; // comp y
                out[2 * nci * ncj + base_idx] += g0x * g0y * g1z; // comp z
            }
        }
    }

    out
}

/// Compute grids-ipip (nabla_i^2) Cartesian integral for one primitive pair.
///
/// Returns 9 * nci * ncj elements (9 = 3x3 second-derivative components).
/// Output layout matches libcint: column-transposed ordering.
fn grids_contract_ipip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IPIP requires G-tensor with li+2 bra levels.
    let nmax = (li + 2 + lj) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 2 + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    // Rys roots/weights sized to nrys_roots (1..=5). The device kernels wire only
    // rys_root{1,2} (GRIDS_MAX_DEVICE_NROOTS); this host fallback handles nroots>2,
    // so the arrays MUST have nrys_roots entries (not a fixed 2) or the loop below
    // indexes out of bounds. rys_roots_host is byte-identical to rys_root{1,2}_host
    // for nroots<=2 and correct for 3..=5 (fail-closed >5 — deferred Wheeler).
    let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // Per-root Rys quadrature: the nabla ladders (g1/g2/g3) and the 9-component
    // products must be formed per root and the PRODUCT accumulated across roots,
    // never the per-axis G-tensor summed first (see grids_contract_ip).
    let mut s = vec![0.0_f64; 9 * nci * ncj];

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

        let mut g0 = vec![0.0_f64; 3 * g_per_axis];
        g0[0] = 1.0;
        g0[g_per_axis] = 1.0;
        g0[2 * g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g0[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[g_per_axis..2 * g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[2 * g_per_axis..3 * g_per_axis], c00[2], rt, nmax, 1);
        }

        if lj >= 1 {
            hrr_step_host(
                &mut g0[0..g_per_axis],
                rirj[0],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
            hrr_step_host(
                &mut g0[g_per_axis..2 * g_per_axis],
                rirj[1],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
            hrr_step_host(
                &mut g0[2 * g_per_axis..3 * g_per_axis],
                rirj[2],
                1,
                nmax + 1,
                nmax,
                lj as u32,
            );
        }

        // g1 = D_i(g0) applied with li+1 bra levels
        let g1 = apply_nabla_i_3axis(&g0, ai, li as u32 + 1, lj as u32, nmax, g_per_axis);
        // g2 = D_i(g0) applied with li bra levels (same as g1 but li not li+1)
        let g2 = apply_nabla_i_3axis(&g0, ai, li as u32, lj as u32, nmax, g_per_axis);
        // g3 = D_i(g1) applied to g1 with li levels
        let g3 = apply_nabla_i_3axis(&g1, ai, li as u32, lj as u32, nmax, g_per_axis);

        // Contract: 9 components
        // libcint ipip layout (from autocode): s[0..8] -> gout with transposition
        // gout[n*9+0] = s0, gout[n*9+1] = s3, gout[n*9+2] = s6, ... (column-transposed)
        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let base = ci_idx * ncj + cj_idx;
                let g0x = g0[jx as usize * dj + ix as usize];
                let g0y = g0[g_per_axis + jy as usize * dj + iy as usize];
                let g0z = g0[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g1x = g1[jx as usize * dj + ix as usize];
                let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
                let g1z = g1[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g2x = g2[jx as usize * dj + ix as usize];
                let g2y = g2[g_per_axis + jy as usize * dj + iy as usize];
                let g2z = g2[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g3x = g3[jx as usize * dj + ix as usize];
                let g3y = g3[g_per_axis + jy as usize * dj + iy as usize];
                let g3z = g3[2 * g_per_axis + jz as usize * dj + iz as usize];

                // s[0..8] = [g3x*g0y*g0z, g2x*g1y*g0z, g2x*g0y*g1z,
                //            g1x*g2y*g0z, g0x*g3y*g0z, g0x*g2y*g1z,
                //            g1x*g0y*g2z, g0x*g1y*g2z, g0x*g0y*g3z]
                let s0 = g3x * g0y * g0z;
                let s1 = g2x * g1y * g0z;
                let s2 = g2x * g0y * g1z;
                let s3 = g1x * g2y * g0z;
                let s4 = g0x * g3y * g0z;
                let s5 = g0x * g2y * g1z;
                let s6 = g1x * g0y * g2z;
                let s7 = g0x * g1y * g2z;
                let s8 = g0x * g0y * g3z;

                // libcint ipip gout[n*9+k]: s0 s3 s6 s1 s4 s7 s2 s5 s8 (column-transposed)
                s[0 * nci * ncj + base] += s0;
                s[1 * nci * ncj + base] += s3;
                s[2 * nci * ncj + base] += s6;
                s[3 * nci * ncj + base] += s1;
                s[4 * nci * ncj + base] += s4;
                s[5 * nci * ncj + base] += s7;
                s[6 * nci * ncj + base] += s2;
                s[7 * nci * ncj + base] += s5;
                s[8 * nci * ncj + base] += s8;
            }
        }
    }

    s
}

/// Compute grids-ipvip (nabla_i x nabla_j) Cartesian integral.
///
/// Returns 9 * nci * ncj elements. Same 9-component layout as ipip but with
/// nabla on both bra (i) and ket (j).
fn grids_contract_ipvip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    aj: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IPVIP requires li+1 bra and lj+1 ket levels.
    let nmax = (li + 1 + lj + 1) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 1 + lj + 1) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 2)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    // Rys roots/weights sized to nrys_roots (1..=5). The device kernels wire only
    // rys_root{1,2} (GRIDS_MAX_DEVICE_NROOTS); this host fallback handles nroots>2,
    // so the arrays MUST have nrys_roots entries (not a fixed 2) or the loop below
    // indexes out of bounds. rys_roots_host is byte-identical to rys_root{1,2}_host
    // for nroots<=2 and correct for 3..=5 (fail-closed >5 — deferred Wheeler).
    let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // Per-root Rys quadrature: the bra/ket nabla ladders (g1/g2/g3) and the
    // 9-component products must be formed per root and the PRODUCT accumulated
    // across roots, never the per-axis G-tensor summed first (see
    // grids_contract_ip).
    let mut s = vec![0.0_f64; 9 * nci * ncj];

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

        let mut g0 = vec![0.0_f64; 3 * g_per_axis];
        g0[0] = 1.0;
        g0[g_per_axis] = 1.0;
        g0[2 * g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g0[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[g_per_axis..2 * g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g0[2 * g_per_axis..3 * g_per_axis], c00[2], rt, nmax, 1);
        }

        // HRR to lj+1 for derivative on ket
        let lj_hrr = lj as u32 + 1;
        hrr_step_host(&mut g0[0..g_per_axis], rirj[0], 1, nmax + 1, nmax, lj_hrr);
        hrr_step_host(
            &mut g0[g_per_axis..2 * g_per_axis],
            rirj[1],
            1,
            nmax + 1,
            nmax,
            lj_hrr,
        );
        hrr_step_host(
            &mut g0[2 * g_per_axis..3 * g_per_axis],
            rirj[2],
            1,
            nmax + 1,
            nmax,
            lj_hrr,
        );

        // g1 = D_j(g0)
        let g1 = apply_nabla_j_3axis(&g0, aj, li as u32 + 1, lj as u32, nmax, g_per_axis);
        // g2 = D_i(g0)
        let g2 = apply_nabla_i_3axis(&g0, ai, li as u32, lj as u32 + 1, nmax, g_per_axis);
        // g3 = D_i(g1)
        let g3 = apply_nabla_i_3axis(&g1, ai, li as u32, lj as u32, nmax, g_per_axis);

        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let base = ci_idx * ncj + cj_idx;
                let g0x = g0[jx as usize * dj + ix as usize];
                let g0y = g0[g_per_axis + jy as usize * dj + iy as usize];
                let g0z = g0[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g1x = g1[jx as usize * dj + ix as usize];
                let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
                let g1z = g1[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g2x = g2[jx as usize * dj + ix as usize];
                let g2y = g2[g_per_axis + jy as usize * dj + iy as usize];
                let g2z = g2[2 * g_per_axis + jz as usize * dj + iz as usize];
                let g3x = g3[jx as usize * dj + ix as usize];
                let g3y = g3[g_per_axis + jy as usize * dj + iy as usize];
                let g3z = g3[2 * g_per_axis + jz as usize * dj + iz as usize];

                let s0 = g3x * g0y * g0z;
                let s1 = g2x * g1y * g0z;
                let s2 = g2x * g0y * g1z;
                let s3 = g1x * g2y * g0z;
                let s4 = g0x * g3y * g0z;
                let s5 = g0x * g2y * g1z;
                let s6 = g1x * g0y * g2z;
                let s7 = g0x * g1y * g2z;
                let s8 = g0x * g0y * g3z;

                s[0 * nci * ncj + base] += s0;
                s[1 * nci * ncj + base] += s1;
                s[2 * nci * ncj + base] += s2;
                s[3 * nci * ncj + base] += s3;
                s[4 * nci * ncj + base] += s4;
                s[5 * nci * ncj + base] += s5;
                s[6 * nci * ncj + base] += s6;
                s[7 * nci * ncj + base] += s7;
                s[8 * nci * ncj + base] += s8;
            }
        }
    }

    s
}

/// Compute grids-spvsp (sigma-p . 1/r . sigma-p) Cartesian integral.
///
/// Returns 4 * nci * ncj elements. Uses the same G-tensor as ipvip but
/// combines the 9 intermediates into 4 spvsp components per libcint autocode.
///
/// spvsp gout: [s5-s7, s6-s2, s1-s3, s0+s4+s8]
fn grids_contract_spvsp(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    aj: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // Same G-tensor as ipvip
    let ipvip = grids_contract_ipvip(pd, ri, rj, rc, ai, aj, li, lj);
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nij = nci * ncj;

    // ipvip s[0..8]: s0 s1 s2 s3 s4 s5 s6 s7 s8
    // spvsp gout: s5-s7, s6-s2, s1-s3, s0+s4+s8
    let mut out = vec![0.0_f64; 4 * nij];

    for k in 0..nij {
        let s0 = ipvip[0 * nij + k];
        let s1 = ipvip[1 * nij + k];
        let s2 = ipvip[2 * nij + k];
        let s3 = ipvip[3 * nij + k];
        let s4 = ipvip[4 * nij + k];
        let s5 = ipvip[5 * nij + k];
        let s6 = ipvip[6 * nij + k];
        let s7 = ipvip[7 * nij + k];
        let s8 = ipvip[8 * nij + k];

        out[0 * nij + k] = s5 - s7;
        out[1 * nij + k] = s6 - s2;
        out[2 * nij + k] = s1 - s3;
        out[3 * nij + k] = s0 + s4 + s8;
    }

    out
}

/// Core grids kernel: compute integrals for all grid points in `grids_params`.
///
/// For each grid point g, computes the integral and writes to:
///   `staging[comp * ngrids * nsi * nsj + j_sph * ngrids * nsi + i_sph * ngrids + g]`
///
/// This matches the libcint c2s_sph_1e_grids output layout where ngrids is the
/// innermost (fastest-varying) index.
fn launch_grids_kernel(
    plan: &ExecutionPlan<'_>,
    grids_params: &GridsEnvParams,
    ncomp: usize,
    staging: &mut [f64],
    contract_fn: impl Fn(
        &crate::math::pdata::PairData,
        [f64; 3],
        [f64; 3],
        [f64; 3],
        f64,
        f64,
        u8,
        u8,
    ) -> Vec<f64>,
) -> Result<(), cintxRsError> {
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;

    let ngrids = grids_params.ngrids;
    let grid_coords = &grids_params.grid_coords;

    if grid_coords.len() < ngrids {
        return Err(cintxRsError::InvalidEnvParam {
            param: "PTR_GRIDS",
            reason: format!(
                "grid_coords length {} < ngrids {}",
                grid_coords.len(),
                ngrids
            ),
        });
    }

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    // For each grid point
    for g in 0..ngrids {
        let rc = grid_coords[g];

        // Accumulate over primitive pairs for this grid point
        let mut cart_buf = vec![0.0_f64; ncomp * nci * ncj];

        for pi in 0..n_prim_i {
            let ai = shell_i.exponents[pi];
            for pj in 0..n_prim_j {
                let aj = shell_j.exponents[pj];

                let pd =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

                let prim_buf = contract_fn(&pd, ri, rj, rc, ai, aj, li, lj);

                // Accumulate over contractions
                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                        let weight = coeff_i * coeff_j;
                        for k in 0..prim_buf.len() {
                            cart_buf[k] += weight * prim_buf[k];
                        }
                    }
                }
            }
        }

        // Apply sp_scale
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_buf.iter_mut() {
                *v *= sp_scale;
            }
        }

        // Transform cart to sph for each component and write to staging
        // Output layout: staging[comp * ngrids * nsi * nsj + j * ngrids * nsi + i * ngrids + g]
        for c in 0..ncomp {
            let cart_comp = &cart_buf[c * nci * ncj..(c + 1) * nci * ncj];
            let mut sph_buf = vec![0.0_f64; nsi * nsj];
            cart_to_sph_1e(cart_comp, &mut sph_buf, li, lj);

            let comp_offset = c * ngrids * nsi * nsj;
            for j_sph in 0..nsj {
                for i_sph in 0..nsi {
                    // libcint layout: out[g + i * ngrids + j * ngrids * ni]
                    let idx = comp_offset + j_sph * ngrids * nsi + i_sph * ngrids + g;
                    staging[idx] = sph_buf[i_sph * nsj + j_sph];
                }
            }
        }
    }

    Ok(())
}

/// Build ExecutionStats for a completed grids kernel call.
fn grids_stats(plan: &ExecutionPlan<'_>, written: usize) -> ExecutionStats {
    let staging_bytes = written * std::mem::size_of::<f64>();
    ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0: written as i32, // conservatively report all as non-zero
        fallback_reason: plan.workspace.fallback_reason,
    }
}

/// Real grids kernel launch function.
///
/// Implements int1e_grids_sph and derivative variants.
/// Output layout matches libcint c2s_sph_1e_grids:
///   staging[comp * ngrids * di * dj + j * ngrids * di + i * ngrids + g]
pub fn launch_grids(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "grids" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_grids",
            detail: format!(
                "canonical_family mismatch: expected grids, got {}",
                specialization.canonical_family()
            ),
        });
    }

    // The scalar nuclear-like arm dispatches its per-(grid, prim-pair) Cartesian
    // G-tensor build to the CubeCL device (run_grids_nuclear_on_backend); the
    // derivative arms (ip/ipip/ipvip/spvsp) stay host (documented split).

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_grids",
            detail: "grids kernel requires exactly 2 shells".to_owned(),
        });
    }

    // Grids params must be populated by the caller (raw compat reads from env[11..12]).
    let grids_params = plan
        .operator_env_params
        .grids_params
        .as_ref()
        .ok_or_else(|| cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            reason: "grids_params not populated — caller must set env[11] and env[12]".to_owned(),
        })?;

    // The env array is not available in the ExecutionPlan; we need it passed through a side
    // channel. For the raw compat path, the kernel receives env through grids_params.ptr_grids
    // which was already adjusted to point at the correct grid range start.
    // We need the actual env array. This is passed as a "basis extension" in the plan.
    // However, the current plan API doesn't carry env directly. We need to access it.
    //
    // Grid coordinates are carried in grids_params.grid_coords, populated by the caller
    // (eval_raw extracts from env[ptr_grids..] at call time before dispatching the plan).

    let op_name = plan.descriptor.operator_name();

    let ngrids = grids_params.ngrids;
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nsi = nsph(li);
    let nsj = nsph(lj);

    match op_name {
        "grids" => {
            let ncomp = 1usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(
                plan,
                grids_params,
                ncomp,
                &mut staging[..required],
                |pd, ri, rj, rc, _ai, _aj, li, lj| {
                    let nci = ncart(li) as u32;
                    let ncj = ncart(lj) as u32;
                    let nmax = (li + lj) as u32;
                    let lj_hrr = lj as u32;
                    let dj = nmax + 1;
                    let g_per_axis = (nmax + 1) * (lj_hrr + 1);
                    let nroots = (li + lj) as u32 / 2 + 1;
                    // Fail-closed: the device kernel wires rys_root{1,2} (nroots<=2),
                    // covering the s/p shells this scalar path targets. Higher-L pairs
                    // fall back to the host reference (byte-identical algorithm).
                    if nroots > GRIDS_MAX_DEVICE_NROOTS {
                        return grids_contract_nuclear_like(pd, ri, rj, rc, li, lj, 0, 0);
                    }
                    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
                    run_grids_nuclear_on_backend(
                        backend, nroots, li as u32, lj as u32, nmax, lj_hrr, dj, g_per_axis, nci,
                        ncj, ri, rj, rc, rp, pd.zeta_ab, pd.aij2, pd.fac,
                    )
                },
            )?;
            Ok(grids_stats(plan, required))
        }
        "grids_ip" => {
            let ncomp = 3usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(
                plan,
                grids_params,
                ncomp,
                &mut staging[..required],
                |pd, ri, rj, rc, ai, aj, li, lj| {
                    let nci = ncart(li) as u32;
                    let ncj = ncart(lj) as u32;
                    let nmax = (li + 1 + lj) as u32;
                    let lj_hrr = lj as u32;
                    let dj = nmax + 1;
                    let g_per_axis = (nmax + 1) * (lj as u32 + 1);
                    let nroots = (li + 1 + lj) as u32 / 2 + 1;
                    // Fail-closed: device wires rys_root{1,2}; higher-L falls back to host.
                    if nroots > GRIDS_MAX_DEVICE_NROOTS {
                        return grids_contract_ip(pd, ri, rj, rc, ai, li, lj);
                    }
                    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
                    run_grids_deriv_on_backend(
                        backend,
                        GRIDS_OP_IP,
                        nroots,
                        li as u32,
                        lj as u32,
                        nmax,
                        lj_hrr,
                        dj,
                        g_per_axis,
                        nci,
                        ncj,
                        3,
                        ri,
                        rj,
                        rc,
                        rp,
                        pd.zeta_ab,
                        pd.aij2,
                        pd.fac,
                        ai,
                        aj,
                    )
                },
            )?;
            Ok(grids_stats(plan, required))
        }
        "grids_ipvip" => {
            let ncomp = 9usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(
                plan,
                grids_params,
                ncomp,
                &mut staging[..required],
                |pd, ri, rj, rc, ai, aj, li, lj| {
                    let nci = ncart(li) as u32;
                    let ncj = ncart(lj) as u32;
                    let nmax = (li + 1 + lj + 1) as u32;
                    let lj_hrr = lj as u32 + 1;
                    let dj = nmax + 1;
                    let g_per_axis = (nmax + 1) * (lj as u32 + 2);
                    let nroots = (li + 1 + lj + 1) as u32 / 2 + 1;
                    if nroots > GRIDS_MAX_DEVICE_NROOTS {
                        return grids_contract_ipvip(pd, ri, rj, rc, ai, aj, li, lj);
                    }
                    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
                    run_grids_deriv_on_backend(
                        backend,
                        GRIDS_OP_IPVIP,
                        nroots,
                        li as u32,
                        lj as u32,
                        nmax,
                        lj_hrr,
                        dj,
                        g_per_axis,
                        nci,
                        ncj,
                        9,
                        ri,
                        rj,
                        rc,
                        rp,
                        pd.zeta_ab,
                        pd.aij2,
                        pd.fac,
                        ai,
                        aj,
                    )
                },
            )?;
            Ok(grids_stats(plan, required))
        }
        "grids_spvsp" => {
            let ncomp = 4usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(
                plan,
                grids_params,
                ncomp,
                &mut staging[..required],
                |pd, ri, rj, rc, ai, aj, li, lj| {
                    let nci = ncart(li) as u32;
                    let ncj = ncart(lj) as u32;
                    let nmax = (li + 1 + lj + 1) as u32;
                    let lj_hrr = lj as u32 + 1;
                    let dj = nmax + 1;
                    let g_per_axis = (nmax + 1) * (lj as u32 + 2);
                    let nroots = (li + 1 + lj + 1) as u32 / 2 + 1;
                    if nroots > GRIDS_MAX_DEVICE_NROOTS {
                        return grids_contract_spvsp(pd, ri, rj, rc, ai, aj, li, lj);
                    }
                    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
                    run_grids_deriv_on_backend(
                        backend,
                        GRIDS_OP_SPVSP,
                        nroots,
                        li as u32,
                        lj as u32,
                        nmax,
                        lj_hrr,
                        dj,
                        g_per_axis,
                        nci,
                        ncj,
                        4,
                        ri,
                        rj,
                        rc,
                        rp,
                        pd.zeta_ab,
                        pd.aij2,
                        pd.fac,
                        ai,
                        aj,
                    )
                },
            )?;
            Ok(grids_stats(plan, required))
        }
        "grids_ipip" => {
            let ncomp = 9usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(
                plan,
                grids_params,
                ncomp,
                &mut staging[..required],
                |pd, ri, rj, rc, ai, aj, li, lj| {
                    let nci = ncart(li) as u32;
                    let ncj = ncart(lj) as u32;
                    let nmax = (li + 2 + lj) as u32;
                    let lj_hrr = lj as u32;
                    let dj = nmax + 1;
                    let g_per_axis = (nmax + 1) * (lj as u32 + 1);
                    let nroots = (li + 2 + lj) as u32 / 2 + 1;
                    if nroots > GRIDS_MAX_DEVICE_NROOTS {
                        return grids_contract_ipip(pd, ri, rj, rc, ai, li, lj);
                    }
                    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
                    run_grids_deriv_on_backend(
                        backend,
                        GRIDS_OP_IPIP,
                        nroots,
                        li as u32,
                        lj as u32,
                        nmax,
                        lj_hrr,
                        dj,
                        g_per_axis,
                        nci,
                        ncj,
                        9,
                        ri,
                        rj,
                        rc,
                        rp,
                        pd.zeta_ab,
                        pd.aij2,
                        pd.fac,
                        ai,
                        aj,
                    )
                },
            )?;
            Ok(grids_stats(plan, required))
        }
        other => Err(cintxRsError::UnsupportedApi {
            requested: format!("grids operator '{}' is not supported", other),
        }),
    }
}

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;
    use cubecl::Runtime;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Derive the device dispatch params + run the device kernel for one
    /// prim-pair + one grid point, mirroring the `launch_grids` scalar closure.
    fn device_grids_block<R: Runtime>(
        client: &ComputeClient<R>,
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rc: [f64; 3],
        li: u8,
        lj: u8,
    ) -> Vec<f64> {
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let nci = ncart(li) as u32;
        let ncj = ncart(lj) as u32;
        let nmax = (li + lj) as u32;
        let lj_hrr = lj as u32;
        let dj = nmax + 1;
        let g_per_axis = (nmax + 1) * (lj_hrr + 1);
        let nroots = (li + lj) as u32 / 2 + 1;
        let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
        run_grids_nuclear_device::<R>(
            client, nroots, li as u32, lj as u32, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ri, rj,
            rc, rp, pd.zeta_ab, pd.aij2, pd.fac,
        )
    }

    fn assert_close(host: &[f64], dev: &[f64], tag: &str) {
        assert_eq!(host.len(), dev.len(), "length mismatch ({tag})");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "grids device/host mismatch ({tag}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    /// Device-vs-host cross-check (CpuRuntime, f64): the on-device
    /// `grids_scalar_kernel` reproduces the host `grids_contract_nuclear_like`
    /// nuclear-like Cartesian block within atol=1e-12 / rtol=1e-10, for shell
    /// pairs (s,s),(p,s),(s,p),(p,p) against two distinct grid points.
    #[test]
    fn test_device_matches_host_grids() {
        let client = cpu_client();
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        for rc in [[0.3_f64, -0.4, 0.8], [-0.7_f64, 0.2, 1.1]] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
                let pd =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                let host = grids_contract_nuclear_like(&pd, ri, rj, rc, li, lj, 0, 0);
                let dev = device_grids_block(&client, ai, aj, ri, rj, rc, li, lj);
                assert_close(&host, &dev, &format!("grids li={li} lj={lj} rc={rc:?}"));
            }
        }
    }

    /// Genericity smoke test: `grids_scalar_kernel` monomorphizes and launches for
    /// `F = f32` (s-s pair), producing finite output.
    #[test]
    fn test_grids_scalar_kernel_generic_f32() {
        let client = cpu_client();
        let g_zero = vec![0.0_f32; 3];
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let rys_zero = vec![0.0_f32; 1];
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let out_zero = vec![0.0_f32; 1];
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));
        grids_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            crate::plane::backend_plane_cube_dim::<cubecl::cpu::CpuRuntime>(),
            unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3) },
            unsafe { ArrayArg::from_raw_parts(u_h, 1) },
            unsafe { ArrayArg::from_raw_parts(w_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0_f32,
            0.0_f32,
            0.6_f32,
            0.5_f32,
            0.7_f32,
            0.3_f32,
            -0.4_f32,
            0.8_f32,
            0.27_f32,
            0.225_f32,
            0.315_f32,
            2.2_f32,
            0.5_f32,
            0.6_f32,
            GRIDS_PIE4 as f32,
            (2.0 * std::f64::consts::PI) as f32,
            0u32,
            0u32,
            0u32,
            0u32,
            1u32,
            1u32,
            1u32,
            1u32,
            1u32,
        );
        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw);
        assert!(
            out[0].is_finite(),
            "grids f32 kernel produced non-finite output"
        );
    }

    /// ROCm device-vs-host parity oracle for the grids scalar nuclear-like path.
    ///
    /// A backend-only cross-check on the real AMD GPU: `grids_scalar_kernel` on
    /// the HIP runtime vs the host `grids_contract_nuclear_like`, over many
    /// randomized shell pairs / grid points. The end-to-end vendor parity now
    /// runs through `eval_raw` in `tests/grids_random_rocm_parity.rs`; this test
    /// remains as a direct kernel-vs-host check. `#[ignore]` +
    /// `CINTX_ROCM_ORACLE=1` gated.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore]
    fn test_grids_device_vs_host_rocm() {
        assert_eq!(
            std::env::var("CINTX_ROCM_ORACLE").as_deref(),
            Ok("1"),
            "ROCm oracle must be invoked with CINTX_ROCM_ORACLE=1 (and CINTX_BACKEND=rocm)."
        );
        let client = cubecl_hip::HipRuntime::client(&Default::default());

        // Deterministic LCG (Numerical Recipes) for reproducible jitter.
        let mut state: u64 = 0x5ec0_ec90_2605_2c10;
        let mut uniform = |lo: f64, hi: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = (state >> 11) as f64 / (1u64 << 53) as f64;
            lo + frac * (hi - lo)
        };

        let mut mismatch_count = 0usize;
        let mut any_nonzero = false;
        let mut cases = 0usize;

        for _ in 0..16 {
            let ai = uniform(0.4, 2.5);
            let aj = uniform(0.4, 2.5);
            let ri = [0.0, 0.0, 0.0];
            let rj = [uniform(-0.8, 0.8), uniform(-0.8, 0.8), uniform(0.2, 1.4)];
            let rc = [uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(-1.0, 1.0)];
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
                let pd =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                let host = grids_contract_nuclear_like(&pd, ri, rj, rc, li, lj, 0, 0);
                let dev = device_grids_block(&client, ai, aj, ri, rj, rc, li, lj);
                assert_eq!(host.len(), dev.len(), "len mismatch li={li} lj={lj}");
                for (&h, &d) in host.iter().zip(dev.iter()) {
                    if (h - d).abs() > 1e-12 + 1e-10 * h.abs() {
                        mismatch_count += 1;
                    }
                    if h.abs() > 1e-18 {
                        any_nonzero = true;
                    }
                }
                cases += 1;
            }
        }

        println!(
            "grids_device_vs_host_rocm: cases={cases} mismatch_count={mismatch_count} any_nonzero={any_nonzero}"
        );
        assert!(
            any_nonzero,
            "grids ROCm: all-zero output — kernel appears stubbed"
        );
        assert_eq!(
            mismatch_count, 0,
            "grids ROCm device-vs-host: {mismatch_count} mismatch(es) at atol=1e-12/rtol=1e-10"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    //  DERIVATIVE device-vs-host (ip / ipip / ipvip / spvsp).
    //  End-to-end vendor parity runs through `eval_raw`
    //  (unstable_source_parity.rs::grids_parity and grids_random_rocm_parity.rs);
    //  these remain as direct device-vs-host kernel cross-checks.
    // ─────────────────────────────────────────────────────────────────────

    /// Per-op device dispatch params, mirroring the `launch_grids` derivative
    /// closures. Returns None when the (li,lj) pair needs nroots>2 (device wires
    /// only rys_root{1,2}; those pairs fall back to host and aren't device-tested).
    fn deriv_params(op: u32, li: u8, lj: u8) -> Option<(u32, u32, u32, u32, u32, u32, u32, u32)> {
        let (nmax, lj_hrr, g_per_axis, ncomp, nroots) = match op {
            GRIDS_OP_IP => {
                let nmax = (li + 1 + lj) as u32;
                (nmax, lj as u32, (nmax + 1) * (lj as u32 + 1), 3u32, (li + 1 + lj) as u32 / 2 + 1)
            }
            GRIDS_OP_IPIP => {
                let nmax = (li + 2 + lj) as u32;
                (nmax, lj as u32, (nmax + 1) * (lj as u32 + 1), 9u32, (li + 2 + lj) as u32 / 2 + 1)
            }
            GRIDS_OP_IPVIP => {
                let nmax = (li + 1 + lj + 1) as u32;
                (nmax, lj as u32 + 1, (nmax + 1) * (lj as u32 + 2), 9u32, (li + 1 + lj + 1) as u32 / 2 + 1)
            }
            _ /* spvsp */ => {
                let nmax = (li + 1 + lj + 1) as u32;
                (nmax, lj as u32 + 1, (nmax + 1) * (lj as u32 + 2), 4u32, (li + 1 + lj + 1) as u32 / 2 + 1)
            }
        };
        if nroots > GRIDS_MAX_DEVICE_NROOTS {
            return None;
        }
        let dj = nmax + 1;
        let nci = ncart(li) as u32;
        let ncj = ncart(lj) as u32;
        Some((nroots, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp))
    }

    /// Run the device derivative kernel for one prim-pair + grid point.
    fn device_deriv_block<R: Runtime>(
        client: &ComputeClient<R>,
        op: u32,
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rc: [f64; 3],
        li: u8,
        lj: u8,
    ) -> Option<Vec<f64>> {
        let (nroots, nmax, lj_hrr, dj, g_per_axis, nci, ncj, ncomp) = deriv_params(op, li, lj)?;
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
        Some(run_grids_deriv_device::<R>(
            client, op, nroots, li as u32, lj as u32, nmax, lj_hrr, dj, g_per_axis, nci, ncj,
            ncomp, ri, rj, rc, rp, pd.zeta_ab, pd.aij2, pd.fac, ai, aj,
        ))
    }

    fn host_deriv_block(
        op: u32,
        pd: &crate::math::pdata::PairData,
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rc: [f64; 3],
        li: u8,
        lj: u8,
    ) -> Vec<f64> {
        match op {
            GRIDS_OP_IP => grids_contract_ip(pd, ri, rj, rc, ai, li, lj),
            GRIDS_OP_IPIP => grids_contract_ipip(pd, ri, rj, rc, ai, li, lj),
            GRIDS_OP_IPVIP => grids_contract_ipvip(pd, ri, rj, rc, ai, aj, li, lj),
            _ => grids_contract_spvsp(pd, ri, rj, rc, ai, aj, li, lj),
        }
    }

    fn check_deriv_op(op: u32, tag: &str) {
        let client = cpu_client();
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let mut tested = 0usize;
        for rc in [[0.3_f64, -0.4, 0.8], [-0.7_f64, 0.2, 1.1]] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
                let dev = match device_deriv_block(&client, op, ai, aj, ri, rj, rc, li, lj) {
                    Some(d) => d,
                    None => continue, // nroots>2 falls back to host; not device-tested
                };
                let pd =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                let host = host_deriv_block(op, &pd, ai, aj, ri, rj, rc, li, lj);
                assert_close(&host, &dev, &format!("{tag} li={li} lj={lj} rc={rc:?}"));
                tested += 1;
            }
        }
        assert!(
            tested > 0,
            "{tag}: no (li,lj) pair exercised the device path"
        );
    }

    #[test]
    fn test_device_matches_host_grids_ip() {
        check_deriv_op(GRIDS_OP_IP, "grids_ip");
    }

    #[test]
    fn test_device_matches_host_grids_ipip() {
        check_deriv_op(GRIDS_OP_IPIP, "grids_ipip");
    }

    #[test]
    fn test_device_matches_host_grids_ipvip() {
        check_deriv_op(GRIDS_OP_IPVIP, "grids_ipvip");
    }

    #[test]
    fn test_device_matches_host_grids_spvsp() {
        check_deriv_op(GRIDS_OP_SPVSP, "grids_spvsp");
    }

    /// f32 genericity smoke: `grids_deriv_kernel` monomorphizes + launches for
    /// `F = f32` across all 4 ops (s-s pair), producing finite output.
    #[test]
    fn test_grids_deriv_kernel_generic_f32() {
        let client = cpu_client();
        for &(op, ncomp) in &[
            (GRIDS_OP_IP, 3usize),
            (GRIDS_OP_IPIP, 9),
            (GRIDS_OP_IPVIP, 9),
            (GRIDS_OP_SPVSP, 4),
        ] {
            // s-s pair derivation (mirrors deriv_params for li=lj=0).
            let (li, lj) = (0u8, 0u8);
            let (nmax, lj_hrr, g_per_axis, nroots) = match op {
                GRIDS_OP_IP => (1u32, 0u32, 2u32 * 1u32, 1u32),
                GRIDS_OP_IPIP => (2u32, 0u32, 3u32 * 1u32, 2u32),
                _ => (2u32, 1u32, 3u32 * 2u32, 2u32),
            };
            let dj = nmax + 1;
            let axis_len = (3 * g_per_axis) as usize;
            let out_len = ncomp; // nci=ncj=1
            let g_zero = vec![0.0_f32; axis_len];
            let g0_h = client.create_from_slice(f32::as_bytes(&g_zero));
            let g1_h = client.create_from_slice(f32::as_bytes(&g_zero));
            let g2_h = client.create_from_slice(f32::as_bytes(&g_zero));
            let g3_h = client.create_from_slice(f32::as_bytes(&g_zero));
            let rys_zero = vec![0.0_f32; nroots as usize];
            let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
            let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
            let out_zero = vec![0.0_f32; out_len];
            let out_h = client.create_from_slice(f32::as_bytes(&out_zero));
            macro_rules! launch_f32 {
                ($nr:expr, $op:expr) => {
                    grids_deriv_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
                        &client,
                        crate::plane::single_cube_count(),
                        crate::plane::backend_plane_cube_dim::<cubecl::cpu::CpuRuntime>(),
                        unsafe { ArrayArg::from_raw_parts(g0_h.clone(), axis_len) },
                        unsafe { ArrayArg::from_raw_parts(g1_h.clone(), axis_len) },
                        unsafe { ArrayArg::from_raw_parts(g2_h.clone(), axis_len) },
                        unsafe { ArrayArg::from_raw_parts(g3_h.clone(), axis_len) },
                        unsafe { ArrayArg::from_raw_parts(u_h.clone(), nroots as usize) },
                        unsafe { ArrayArg::from_raw_parts(w_h.clone(), nroots as usize) },
                        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                        0.0_f32,
                        0.0_f32,
                        0.0_f32,
                        0.6_f32,
                        0.5_f32,
                        0.7_f32,
                        0.3_f32,
                        -0.4_f32,
                        0.8_f32,
                        0.27_f32,
                        0.225_f32,
                        0.315_f32,
                        2.2_f32,
                        0.5_f32,
                        0.6_f32,
                        0.9_f32,
                        1.3_f32,
                        GRIDS_PIE4 as f32,
                        (2.0 * std::f64::consts::PI) as f32,
                        li as u32,
                        lj as u32,
                        nmax,
                        lj_hrr,
                        dj,
                        g_per_axis,
                        1u32,
                        1u32,
                        $nr,
                        $op,
                    )
                };
            }
            match (nroots, op) {
                (1, GRIDS_OP_IP) => launch_f32!(1u32, GRIDS_OP_IP),
                (2, GRIDS_OP_IPIP) => launch_f32!(2u32, GRIDS_OP_IPIP),
                (2, GRIDS_OP_IPVIP) => launch_f32!(2u32, GRIDS_OP_IPVIP),
                (2, GRIDS_OP_SPVSP) => launch_f32!(2u32, GRIDS_OP_SPVSP),
                _ => unreachable!("unexpected (nroots,op) in f32 smoke"),
            }
            let raw = client.read_one_unchecked(out_h);
            let out = f32::from_bytes(&raw);
            for (i, v) in out.iter().enumerate() {
                assert!(v.is_finite(), "grids deriv f32 op={op} comp={i} non-finite");
            }
        }
    }

    /// ROCm device-vs-host parity oracle for the grids DERIVATIVE paths
    /// (ip / ipip / ipvip / spvsp). A direct device-vs-host cross-check on the
    /// real AMD GPU (`run_grids_deriv_device::<HipRuntime>` vs host
    /// `grids_contract_<op>`); end-to-end vendor parity now runs through
    /// `eval_raw` in `tests/grids_random_rocm_parity.rs`. `#[ignore]` +
    /// `CINTX_ROCM_ORACLE=1` gated.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore]
    fn test_grids_deriv_device_vs_host_rocm() {
        assert_eq!(
            std::env::var("CINTX_ROCM_ORACLE").as_deref(),
            Ok("1"),
            "ROCm oracle must be invoked with CINTX_ROCM_ORACLE=1 (and CINTX_BACKEND=rocm)."
        );
        let client = cubecl_hip::HipRuntime::client(&Default::default());

        let mut state: u64 = 0x9a00_2605_3009_1234u64.wrapping_add(0xdead_beef);
        let mut uniform = |lo: f64, hi: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = (state >> 11) as f64 / (1u64 << 53) as f64;
            lo + frac * (hi - lo)
        };

        for &(op, tag) in &[
            (GRIDS_OP_IP, "grids_ip"),
            (GRIDS_OP_IPIP, "grids_ipip"),
            (GRIDS_OP_IPVIP, "grids_ipvip"),
            (GRIDS_OP_SPVSP, "grids_spvsp"),
        ] {
            let mut mismatch_count = 0usize;
            let mut any_nonzero = false;
            let mut cases = 0usize;
            for _ in 0..16 {
                let ai = uniform(0.4, 2.5);
                let aj = uniform(0.4, 2.5);
                let ri = [0.0, 0.0, 0.0];
                let rj = [uniform(-0.8, 0.8), uniform(-0.8, 0.8), uniform(0.2, 1.4)];
                let rc = [uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(-1.0, 1.0)];
                for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
                    let dev = match device_deriv_block(&client, op, ai, aj, ri, rj, rc, li, lj) {
                        Some(d) => d,
                        None => continue,
                    };
                    let pd = compute_pdata_host(
                        ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                    );
                    let host = host_deriv_block(op, &pd, ai, aj, ri, rj, rc, li, lj);
                    assert_eq!(host.len(), dev.len(), "{tag} len mismatch li={li} lj={lj}");
                    for (&h, &d) in host.iter().zip(dev.iter()) {
                        if (h - d).abs() > 1e-12 + 1e-10 * h.abs() {
                            mismatch_count += 1;
                        }
                        if h.abs() > 1e-18 {
                            any_nonzero = true;
                        }
                    }
                    cases += 1;
                }
            }
            println!(
                "grids_deriv_device_vs_host_rocm[{tag}]: cases={cases} mismatch_count={mismatch_count} any_nonzero={any_nonzero}"
            );
            assert!(cases > 0, "{tag} ROCm: no device-path cases exercised");
            assert!(
                any_nonzero,
                "{tag} ROCm: all-zero output — kernel appears stubbed"
            );
            assert_eq!(
                mismatch_count, 0,
                "{tag} ROCm device-vs-host: {mismatch_count} mismatch(es) at atol=1e-12/rtol=1e-10"
            );
        }
    }
}
