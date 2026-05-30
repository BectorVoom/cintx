//! 3c1e (three-center one-electron overlap) integral kernel.
//!
//! Implements the G-tensor fill + contraction + cart-to-sph pipeline from
//! libcint `g3c1e.c` (`CINTg3c1e_ovlp`) and `cint3c1e.c` (`CINT3c1e_loop_nopt`).
//!
//! # Execution model (CubeCL device dispatch)
//!
//! The numeric core — the per-shell-triple G-tensor fill (combined j+k VRR →
//! i-HRR → k-separation HRR) and the triple Cartesian contraction — runs as a
//! real CubeCL `#[cube(launch)]` kernel ([`center_3c1e_kernel`]) **generic over
//! `F: Float`**, dispatched onto the resolved backend's `ComputeClient` (CPU
//! `CpuRuntime`, ROCm `HipRuntime`, …) via [`run_3c1e_device`]. The Cartesian
//! buffer is read back to the host and the `cart_to_sph` transform (whose
//! coefficient tables are host-only) finishes on the host.
//!
//! Unlike 2c2e, 3c1e has **no Rys quadrature** — this is a three-center Gaussian
//! product overlap, so the whole G-tensor fill + contraction is inlined directly
//! in the kernel body.
//!
//! ## Precision policy
//!
//! The kernel is genuinely generic over `F: Float`, but the launcher runs it at
//! **f64** on-device for both `PrecisionKind` variants and casts the read-back
//! buffer to `F` at the c2s/output stage via `F::from_f64_lossy`. This preserves
//! the historical "intermediates in f64, output cast to `F`" contract that the
//! f32 parity gate is calibrated against, while moving the real arithmetic onto
//! the device.
//!
//! # Algorithm
//! For each contracted shell triple (i, j, k):
//! 1. Compute per-primitive prefactor from three-center Gaussian overlap exponent.
//! 2. Fill G-tensor via VRR in combined (j+k) direction (NEGATIVE-sign disp),
//!    then HRR to split i and k.
//! 3. Contract Cartesian components weighted by ci*cj*ck contraction coefficients.
//! 4. Accumulate over all primitive triples (kp, jp, ip).
//! 5. Apply `common_fac_sp` scaling for s/p shells (folded into common_factor).
//! 6. Apply cart-to-sph transform if representation is Spheric.
//!
//! # G-tensor layout (from CINTinit_int3c1e_EnvVars)
//! ```text
//! dli = li + 1,  dlj = lj + lk + 1,  dlk = lk + 1
//! g_stride_i = 1
//! g_stride_j = dli
//! g_stride_k = dli * dlj
//! g_size = max(dli*dlj*dlk, dli*vrr_nmax),  vrr_nmax = li + lj + lk + 1
//! ```

use crate::backend::ResolvedBackend;
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c1e, ncart, nsph};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Matches `CINTcommon_fac_sp` from libcint `cart2sph.c`:
///   - l=0: 0.282094791773878 = 1/(2*sqrt(pi)) = Y_0^0
///   - l=1: 0.488602511902920 = sqrt(3/(4*pi))
///   - l>=2: 1.0
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
/// the host-vs-device cross-check and the G-tensor unit tests, and reused by the
/// Phase 23 host-side ip1/iprinv gradient contraction.
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

/// 3c1e G-tensor fill + triple Cartesian contraction for one shell triple,
/// on-device.
///
/// Single work item (`UNIT_POS == 0`) — a faithful, correctness-first port of
/// the host pipeline (`fill_g_tensor_3c1e` + `contract_3c1e_ovlp`). It is not
/// yet parallelized across primitives/components; that is future work.
///
/// No Rys quadrature, so the entire G-tensor fill (base case → combined-(j+k)
/// VRR with NEGATIVE displacement → i-HRR → k-separation HRR) and the triple
/// Cartesian contraction are inlined here. All math uses `F::exp` / `F::sqrt` /
/// `F::cast_from` / `F::new` (never methods), statement-form `if`, and `u32`
/// indices/counters with `while` loops — never host `for`/`Vec`/`continue`.
///
/// The kernel iterates the full primitive triple loop (kp, jp, ip), applies the
/// per-primitive contraction coefficient product `coeff_i[ip]*coeff_j[jp]*
/// coeff_k[kp]` to each primitive's contribution, and accumulates into
/// `cart_out`. It therefore evaluates one (ci, cj, ck) contraction triple per
/// launch (the common nctr==1 case is a single launch).
///
/// Layout of `g` (size `3 * g_alloc`): `[gx | gy | gz]`, each `g_alloc` long.
/// `g_alloc = max(dli*dlj*dlk, dli*vrr_nmax)` with `dli=li+1`, `dlj=lj+lk+1`,
/// `dlk=lk+1`, `vrr_nmax=li+lj+lk+1`. `g_stride_i=1`, `g_stride_j=dli`,
/// `g_stride_k=dli*dlj`.
///
/// `cart_out` (size `nci*ncj*nck`) is zeroed in-kernel and accumulated over all
/// primitive triples and contraction components:
/// `cart_out[(k_idx*ncj + j_idx)*nci + i_idx]` (i fastest, k slowest).
///
/// Source: libcint-master/src/g3c1e.c `CINTg3c1e_ovlp`,
///         cint3c1e.c `CINT3c1e_loop_nopt`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn center_3c1e_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    exps_k: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    coeff_k: &Array<F>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    rkx: F,
    rky: F,
    rkz: F,
    common_factor: F,
    expcutoff: F,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
) {
    if UNIT_POS == 0u32 {
        let dli = li + 1u32;
        let dlj = lj + lk + 1u32; // combined j+k dimension
        let dlk = lk + 1u32;

        // VRR length in the combined dimension and the k-separation HRR bound.
        let nmax = li + lj + lk; // total VRR length in combined dimension
        let mmax = lj + lk; // used for k-separation HRR bound
        let vrr_nmax = li + lj + lk + 1u32; // = li + dlj

        // g_alloc = max(dli*dlj*dlk, dli*vrr_nmax)
        let prod = dli * dlj * dlk;
        let vrr_buf = dli * vrr_nmax;
        let mut g_alloc = prod;
        if vrr_buf > g_alloc {
            g_alloc = vrr_buf;
        }
        let total_g = 3u32 * g_alloc;

        // Final-layout strides (also the local VRR stride dj_local = dli).
        let dj = dli; // g_stride_j = dli
        let dk = dli * dlj; // g_stride_k

        // Cartesian output dimensions.
        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
        let out_len = nci * ncj * nck;

        // Contraction indices: this launch evaluates the (0,0,0) contraction
        // triple. nctr_* are carried for signature symmetry; the host launcher
        // dispatches one launch per (ci,cj,ck) with the matching coeff columns.
        let ci_sel = 0u32;
        let cj_sel = 0u32;
        let ck_sel = 0u32;

        // Zero the accumulation buffer.
        let mut oi = 0u32;
        while oi < out_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // Pairwise displacements / squared distances (independent of primitives).
        let xij = rix - rjx;
        let yij = riy - rjy;
        let zij = riz - rjz;
        let xik = rix - rkx;
        let yik = riy - rky;
        let zik = riz - rkz;
        let xjk = rjx - rkx;
        let yjk = rjy - rky;
        let zjk = rjz - rkz;
        let rr_ij = xij * xij + yij * yij + zij * zij;
        let rr_ik = xik * xik + yik * yik + zik * zik;
        let rr_jk = xjk * xjk + yjk * yjk + zjk * zjk;

        // rirj = ri - rj (i-HRR shift).
        let rirjx = xij;
        let rirjy = yij;
        let rirjz = zij;
        // rjrk = rj - rk (k-separation HRR shift).
        let rjrkx = xjk;
        let rjrky = yjk;
        let rjrkz = zjk;

        // Primitive loops: kp outer, then jp, then ip (matching CINT3c1e_loop_nopt).
        let mut kp = 0u32;
        while kp < nprim_k {
            let ak = exps_k[kp as usize];
            let ck_coeff = coeff_k[(kp * nctr_k + ck_sel) as usize];
            let mut jp = 0u32;
            while jp < nprim_j {
                let aj = exps_j[jp as usize];
                let cj_coeff = coeff_j[(jp * nctr_j + cj_sel) as usize];
                let mut ip = 0u32;
                while ip < nprim_i {
                    let ai = exps_i[ip as usize];
                    let ci_coeff = coeff_i[(ip * nctr_i + ci_sel) as usize];
                    let aijk = ai + aj + ak;

                    // Exponential screening: eijk = (ai*aj*rr_ij + ai*ak*rr_ik
                    //   + aj*ak*rr_jk) / aijk
                    let aiajrr = ai * aj * rr_ij;
                    let aiakrr = ai * ak * rr_ik;
                    let ajakrr = aj * ak * rr_jk;
                    let eijk = (aiajrr + aiakrr + ajakrr) / aijk;

                    // No `continue`: guard the primitive body, always advance ip.
                    if eijk <= expcutoff {
                        // Per-primitive prefactor including contraction coeffs:
                        //   dijk = exp(-eijk) / (aijk * sqrt(aijk))
                        //   fac  = common_factor * dijk * ci*cj*ck
                        let weight = ci_coeff * cj_coeff * ck_coeff;
                        let dijk = F::exp(-eijk) / (aijk * F::sqrt(aijk));
                        let fac = common_factor * dijk * weight;

                        let aijk1 = F::new(0.5) / aijk;

                        // ── Fill the G-tensor ─────────────────────────────────
                        // Zero the whole g buffer.
                        let mut gi = 0u32;
                        while gi < total_g {
                            g[gi as usize] = F::new(0.0);
                            gi += 1u32;
                        }

                        // Base case: gx[0]=1, gy[0]=1, gz[0]=fac.
                        g[0u32 as usize] = F::new(1.0);
                        g[g_alloc as usize] = F::new(1.0);
                        g[(2u32 * g_alloc) as usize] = fac;

                        // rjrijk = rj - (ai*ri + aj*rj + ak*rk)/aijk, per axis.
                        // VRR displacement is the NEGATIVE of this.
                        let gx_w = (ai * rix + aj * rjx + ak * rkx) / aijk;
                        let gy_w = (ai * riy + aj * rjy + ak * rky) / aijk;
                        let gz_w = (ai * riz + aj * rjz + ak * rkz) / aijk;
                        let rjrijkx = rjx - gx_w;
                        let rjrijky = rjy - gy_w;
                        let rjrijkz = rjz - gz_w;

                        // VRR + i-HRR + k-separation HRR per axis.
                        let mut axis = 0u32;
                        while axis < 3u32 {
                            let off = axis * g_alloc;

                            // Axis displacement disp = -rjrijk[axis].
                            let mut disp = -rjrijkx;
                            if axis == 1u32 {
                                disp = -rjrijky;
                            }
                            if axis == 2u32 {
                                disp = -rjrijkz;
                            }
                            // Axis i-HRR shift rirj[axis].
                            let mut rirj_d = rirjx;
                            if axis == 1u32 {
                                rirj_d = rirjy;
                            }
                            if axis == 2u32 {
                                rirj_d = rirjz;
                            }
                            // Axis k-separation HRR shift rjrk[axis].
                            let mut rjrk_d = rjrkx;
                            if axis == 1u32 {
                                rjrk_d = rjrky;
                            }
                            if axis == 2u32 {
                                rjrk_d = rjrkz;
                            }

                            // VRR over combined j+k dimension (NEGATIVE-sign disp):
                            //   g[dj]       = disp * g[0]
                            //   g[(j+1)*dj] = aijk1*j*g[(j-1)*dj] + disp*g[j*dj]
                            if nmax >= 1u32 {
                                g[(off + dj) as usize] = disp * g[off as usize];
                                let mut j = 1u32;
                                while j < nmax {
                                    let hi = aijk1
                                        * F::cast_from(j)
                                        * g[(off + (j - 1u32) * dj) as usize]
                                        + disp * g[(off + j * dj) as usize];
                                    g[(off + (j + 1u32) * dj) as usize] = hi;
                                    j += 1u32;
                                }
                            }

                            // i-HRR: shift angular momentum from combined-j to i.
                            //   for i=1..=li, j=0..=nmax-i:
                            //     g[i + j*dj] = g[i-1 + (j+1)*dj] - rirj*g[i-1 + j*dj]
                            if li >= 1u32 {
                                let mut i = 1u32;
                                while i <= li {
                                    let j_max = nmax - i;
                                    let mut j = 0u32;
                                    while j <= j_max {
                                        let idx_out = i + j * dj;
                                        let idx_hi = (i - 1u32) + (j + 1u32) * dj;
                                        let idx_lo = (i - 1u32) + j * dj;
                                        g[(off + idx_out) as usize] = g[(off + idx_hi) as usize]
                                            - rirj_d * g[(off + idx_lo) as usize];
                                        j += 1u32;
                                    }
                                    i += 1u32;
                                }
                            }

                            // k-separation HRR: split combined j+k into j and k.
                            //   for k=1..=lk, j=0..=mmax-k, i=0..=li:
                            //     base = k*dk + j*dj
                            //     g[base+i] = g[base+i + dj - dk] + rjrk*g[base+i - dk]
                            if lk >= 1u32 {
                                let mut k = 1u32;
                                while k <= lk {
                                    let j_max = mmax - k;
                                    let mut j = 0u32;
                                    while j <= j_max {
                                        let base = k * dk + j * dj;
                                        let mut i = 0u32;
                                        while i <= li {
                                            let idx = base + i;
                                            let idx_hi = idx + dj - dk;
                                            let idx_lo = idx - dk;
                                            g[(off + idx) as usize] = g[(off + idx_hi) as usize]
                                                + rjrk_d * g[(off + idx_lo) as usize];
                                            i += 1u32;
                                        }
                                        j += 1u32;
                                    }
                                    k += 1u32;
                                }
                            }

                            axis += 1u32;
                        }

                        let gx = 0u32;
                        let gy = g_alloc;
                        let gz = 2u32 * g_alloc;

                        // ── Contract over Cartesian triples (i fastest, k slowest) ─
                        // Reproduce `cart_comps` ordering inline (lx desc, ly desc)
                        // for each of i/j/k, tracking each linear *_idx.
                        let mut ck_idx = 0u32;
                        let mut ka = 0u32;
                        while ka <= lk {
                            let kx = lk - ka; // descending
                            let lk_minus_kx = lk - kx;
                            let mut kb = 0u32;
                            while kb <= lk_minus_kx {
                                let ky = lk_minus_kx - kb;
                                let kz = lk - kx - ky;

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

                                                let vx = g[(gx + ix + jx * dj + kx * dk) as usize];
                                                let vy = g[(gy + iy + jy * dj + ky * dk) as usize];
                                                let vz = g[(gz + iz + jz * dj + kz * dk) as usize];
                                                let out_idx =
                                                    (ck_idx * ncj + cj_idx) * nci + ci_idx;
                                                cart_out[out_idx as usize] += vx * vy * vz;

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
                    }

                    ip += 1u32;
                }
                jp += 1u32;
            }
            kp += 1u32;
        }
    }
}

/// Dispatch [`center_3c1e_kernel`] at `f64` on a resolved backend's client and
/// read back the Cartesian accumulation buffer (`nci*ncj*nck`, i fastest).
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (see module-level precision policy). The kernel runs
/// the full primitive triple loop and applies the contraction coefficients
/// internally, so one launch produces a fully-contracted (single-nctr) buffer.
#[allow(clippy::too_many_arguments)]
fn run_3c1e_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    common_factor: f64,
    expcutoff: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let lk_u = lk as usize;
    let dli = li_u + 1;
    let dlj = lj_u + lk_u + 1;
    let dlk = lk_u + 1;
    let vrr_nmax = li_u + lj_u + lk_u + 1;
    let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let nck = (lk_u + 1) * (lk_u + 2) / 2;
    let out_len = nci * ncj * nck;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));

    // Scratch + output buffers (zero-initialised on the host for determinism;
    // the kernel also zeros `g` and `cart_out` before use).
    let g_zero = vec![0.0_f64; 3 * g_alloc];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    center_3c1e_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h, exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_j_h, exps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_k_h, exps_k.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h, coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_alloc) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        ri[0],
        ri[1],
        ri[2],
        rj[0],
        rj[1],
        rj[2],
        rk[0],
        rk[1],
        rk[2],
        common_factor,
        expcutoff,
        li,
        lj,
        lk,
        nprim_i,
        nprim_j,
        nprim_k,
        1u32,
        1u32,
        1u32,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Fill the G-tensor for a 3c1e overlap primitive triple.
///
/// Host f64 reference of the exact device algorithm — used by the in-crate unit
/// tests and the host-vs-device cross-check. Implements `CINTg3c1e_ovlp` from
/// `libcint-master/src/g3c1e.c`.
///
/// Returned as flat `3 * g_alloc` array: `[gx | gy | gz]`.
///
/// Phase 23: promoted out of `#[cfg(test)]` — the host-side int3c1e_ip1
/// (overlap) gradient path builds this base at `li+1` headroom.
fn fill_g_tensor_3c1e(
    fac: f64,
    ai: f64,
    aj: f64,
    ak: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rirj: [f64; 3],
    li: u32,
    lj: u32,
    lk: u32,
) -> Vec<f64> {
    let dli = (li + 1) as usize;
    let dlj = (lj + lk + 1) as usize; // combined j+k dimension
    let dlk = (lk + 1) as usize;

    let nmax = (li + lj + lk) as usize; // total VRR length in combined dimension
    let mmax = (lj + lk) as usize; // used for k-separation HRR bound

    let vrr_nmax = li as usize + dlj; // = li + lj + lk + 1
    let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

    let mut g = vec![0.0_f64; 3 * g_alloc];

    let aijk = ai + aj + ak;
    let aijk1 = 0.5_f64 / aijk;

    let gx_off = 0usize;
    let gy_off = g_alloc;
    let gz_off = 2 * g_alloc;

    g[gx_off] = 1.0;
    g[gy_off] = 1.0;
    g[gz_off] = fac;

    if nmax == 0 {
        return g;
    }

    let dj_local = dli; // = li + 1

    let rjrijk = [
        rj[0] - (ai * ri[0] + aj * rj[0] + ak * rk[0]) / aijk,
        rj[1] - (ai * ri[1] + aj * rj[1] + ak * rk[1]) / aijk,
        rj[2] - (ai * ri[2] + aj * rj[2] + ak * rk[2]) / aijk,
    ];

    for d in 0..3 {
        let off = d * g_alloc;
        let disp = -rjrijk[d]; // negative sign from the formula
        g[off + dj_local] = disp * g[off];
        let mut j = 1usize;
        while j < nmax {
            g[off + (j + 1) * dj_local] =
                aijk1 * j as f64 * g[off + (j - 1) * dj_local] + disp * g[off + j * dj_local];
            j += 1;
        }
    }

    for d in 0..3 {
        let off = d * g_alloc;
        let rirj_d = rirj[d]; // = ri[d] - rj[d]
        for i in 1..=(li as usize) {
            let j_max = nmax - i;
            for j in 0..=j_max {
                let idx_out = i + j * dj_local;
                let idx_hi = (i - 1) + (j + 1) * dj_local;
                let idx_lo = (i - 1) + j * dj_local;
                g[off + idx_out] = g[off + idx_hi] - rirj_d * g[off + idx_lo];
            }
        }
    }

    let dk = dli * dlj; // = g_stride_k
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];

    for d in 0..3 {
        let off = d * g_alloc;
        let rjrk_d = rjrk[d];
        for k in 1..=(lk as usize) {
            for j in 0..=(mmax - k) {
                let base = k * dk + j * dj_local;
                for i in 0..=li as usize {
                    let idx = base + i;
                    let idx_hi = idx + dj_local - dk;
                    let idx_lo = idx - dk;
                    g[off + idx] = g[off + idx_hi] + rjrk_d * g[off + idx_lo];
                }
            }
        }
    }

    g
}

/// Fill the G-tensor for a 3c1e **nuclear** (rinv-Coulomb) primitive triple at a
/// single Rys root — the genuinely-new base kernel for `int3c1e_iprinv`.
///
/// This is `fill_g_tensor_3c1e` (the overlap base) EXTENDED with the Rys `t2`
/// parameter, ported from libcint `CINTg3c1e_nuc` (`g3c1e.c:192-235`):
///   - base: `gx[0]=gy[0]=1`, `gz[0] = 2/SQRTPI * fac` (fac = per-root weight).
///   - `aijk1 = 0.5 * (1 - t2) / aijk`  (the overlap base used `0.5/aijk`, t2=0).
///   - `rjr0[d] = rj[d] - (rijk[d] + t2 * (cr[d] - rijk[d]))`,  cr = rinv origin.
///   - VRR: `gx[dj] = -rjr0[0]*gx[0]`, then
///          `gx[(j+1)*dj] = aijk1*j*gx[(j-1)*dj] - rjr0[0]*gx[j*dj]`.
/// followed by the SAME i-HRR (`rirj`) and k-separation HRR (`rjrk`) as the
/// overlap base. At t2=0 / fac scaled this reduces exactly to the overlap fill.
///
/// `rijk = (ai*ri + aj*rj + ak*rk)/aijk` is passed precomputed (the Rys driver
/// also uses it for `x`). `cr` is the rinv origin (`env[PTR_RINV_ORIG..+3]`).
/// Returned as a flat `3 * g_alloc` array `[gx | gy | gz]`.
#[allow(clippy::too_many_arguments)]
fn fill_g_tensor_3c1e_nuc(
    fac: f64,
    t2: f64,
    ai: f64,
    aj: f64,
    ak: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rijk: [f64; 3],
    cr: [f64; 3],
    li: u32,
    lj: u32,
    lk: u32,
) -> Vec<f64> {
    let dli = (li + 1) as usize;
    let dlj = (lj + lk + 1) as usize; // combined j+k dimension
    let dlk = (lk + 1) as usize;

    let nmax = (li + lj + lk) as usize; // total VRR length in combined dimension
    let mmax = (lj + lk) as usize; // k-separation HRR bound

    let vrr_nmax = li as usize + dlj; // = li + lj + lk + 1
    let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

    let mut g = vec![0.0_f64; 3 * g_alloc];

    let aijk = ai + aj + ak;
    let aijk1 = 0.5_f64 * (1.0 - t2) / aijk;

    let gx_off = 0usize;
    let gy_off = g_alloc;
    let gz_off = 2 * g_alloc;

    g[gx_off] = 1.0;
    g[gy_off] = 1.0;
    // 2/SQRTPI folds the nuclear Rys normalization into gz (g3c1e.c:201).
    g[gz_off] = (2.0 / SQRTPI) * fac;

    if nmax == 0 {
        return g;
    }

    let dj_local = dli; // = li + 1

    // rjr0[d] = rj[d] - (rijk[d] + t2*(cr[d] - rijk[d])).
    let rjr0 = [
        rj[0] - (rijk[0] + t2 * (cr[0] - rijk[0])),
        rj[1] - (rijk[1] + t2 * (cr[1] - rijk[1])),
        rj[2] - (rijk[2] + t2 * (cr[2] - rijk[2])),
    ];

    for d in 0..3 {
        let off = d * g_alloc;
        // gx[dj] = -rjr0[d] * gx[0]
        g[off + dj_local] = -rjr0[d] * g[off];
        let mut j = 1usize;
        while j < nmax {
            g[off + (j + 1) * dj_local] = aijk1 * j as f64 * g[off + (j - 1) * dj_local]
                - rjr0[d] * g[off + j * dj_local];
            j += 1;
        }
    }

    // i-HRR (shift combined-j → i), identical to the overlap base.
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    for d in 0..3 {
        let off = d * g_alloc;
        let rirj_d = rirj[d];
        for i in 1..=(li as usize) {
            let j_max = nmax - i;
            for j in 0..=j_max {
                let idx_out = i + j * dj_local;
                let idx_hi = (i - 1) + (j + 1) * dj_local;
                let idx_lo = (i - 1) + j * dj_local;
                g[off + idx_out] = g[off + idx_hi] - rirj_d * g[off + idx_lo];
            }
        }
    }

    // k-separation HRR, identical to the overlap base.
    let dk = dli * dlj;
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    for d in 0..3 {
        let off = d * g_alloc;
        let rjrk_d = rjrk[d];
        for k in 1..=(lk as usize) {
            for j in 0..=(mmax - k) {
                let base = k * dk + j * dj_local;
                for i in 0..=li as usize {
                    let idx = base + i;
                    let idx_hi = idx + dj_local - dk;
                    let idx_lo = idx - dk;
                    g[off + idx] = g[off + idx_hi] + rjrk_d * g[off + idx_lo];
                }
            }
        }
    }

    g
}

/// Apply the bra-i 1e nabla to a 3c1e G-tensor (`CINTnabla1i_3c1e`, g3c1e.c:262).
///
/// The G-tensor `g` is built with `li+1` headroom (so the dli stride is `li+2`).
/// Produces `g1` of the same flat `3*g_alloc` layout with `∂χ_i` applied per axis:
///   ix == 0:  g1[ptr] = -2*ai * g[ptr+1]
///   ix >= 1:  g1[ptr+i] = i*g[ptr+i-1] - 2*ai*g[ptr+i+1]
/// over every (j,k) slice, where `ptr = dj_h*j + dk_h*k` and `dj_h = li_ceil+1`,
/// `dk_h = dj_h*(lj+lk+1)` are the HEADROOM strides (li_ceil = li+1).
fn nabla1i_3c1e(g: &[f64], li: u32, lj: u32, lk: u32, ai: f64) -> Vec<f64> {
    let li_ceil = (li + 1) as usize;
    let dli_h = li_ceil + 1; // dj stride of the headroom g-tensor
    let dlj_h = (lj + lk + 1) as usize;
    let dlk_h = (lk + 1) as usize;
    let vrr_nmax = li_ceil + dlj_h;
    let g_alloc = (dli_h * dlj_h * dlk_h).max(dli_h * vrr_nmax);

    let dj_h = dli_h; // g_stride_j
    let dk_h = dli_h * dlj_h; // g_stride_k
    let ai2 = -2.0 * ai;

    let mut g1 = vec![0.0_f64; 3 * g_alloc];
    for d in 0..3usize {
        let off = d * g_alloc;
        for k in 0..=(lk as usize) {
            for j in 0..=(lj as usize) {
                let ptr = dj_h * j + dk_h * k;
                g1[off + ptr] = ai2 * g[off + ptr + 1];
                for i in 1..=(li as usize) {
                    g1[off + ptr + i] = i as f64 * g[off + ptr + i - 1] + ai2 * g[off + ptr + i + 1];
                }
            }
        }
    }
    g1
}

/// 3-component bra-i gradient contraction for 3c1e (ip1 / iprinv share this).
///
/// `g` (g0) and `g1` (= `nabla1i_3c1e(g0)`) are the headroom G-tensors (built at
/// `li+1`). Produces the component-leading Cartesian output
/// `out[comp * nci*ncj*nck + (k_idx*ncj + j_idx)*nci + i_idx]` for comp in 0..3:
///   comp 0 (∂/∂Ax): g1x·g0y·g0z
///   comp 1 (∂/∂Ay): g0x·g1y·g0z
///   comp 2 (∂/∂Az): g0x·g0y·g1z
/// (libcint `CINTgout1e_int3c1e_ip1`/`_iprinv`, int3c1e.c:78/133 — same gout).
fn contract_3c1e_grad(g: &[f64], g1: &[f64], li: u8, lj: u8, lk: u8) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);

    // Headroom strides (li raised by 1 when g was built).
    let dli_h = (li as usize) + 2; // (li+1)+1
    let dlj_h = (lj as usize) + (lk as usize) + 1;
    let dlk_h = (lk as usize) + 1;
    let li_ceil = (li as usize) + 1;
    let vrr_nmax = li_ceil + dlj_h;
    let g_alloc = (dli_h * dlj_h * dlk_h).max(dli_h * vrr_nmax);

    let dj = dli_h; // g_stride_j
    let dk = dli_h * dlj_h; // g_stride_k

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);

    let gx = 0usize;
    let gy = g_alloc;
    let gz = 2 * g_alloc;

    let block_len = nci * ncj * nck;
    let mut out = vec![0.0_f64; 3 * block_len];

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let nx = ix as usize + jx as usize * dj + kx as usize * dk;
                let ny = iy as usize + jy as usize * dj + ky as usize * dk;
                let nz = iz as usize + jz as usize * dj + kz as usize * dk;

                let g0x = g[gx + nx];
                let g0y = g[gy + ny];
                let g0z = g[gz + nz];
                let g1x = g1[gx + nx];
                let g1y = g1[gy + ny];
                let g1z = g1[gz + nz];

                let n = (k_idx * ncj + j_idx) * nci + i_idx;
                out[n] += g1x * g0y * g0z;
                out[block_len + n] += g0x * g1y * g0z;
                out[2 * block_len + n] += g0x * g0y * g1z;
            }
        }
    }

    out
}

/// Contract G-tensor for 3c1e overlap operator (host f64 reference).
///
/// Output layout: i fastest (innermost), k slowest (outermost):
/// `out[(k_idx*ncj + j_idx)*nci + i_idx]`.
#[cfg(test)]
fn contract_3c1e_ovlp(g: &[f64], li: u8, lj: u8, lk: u8, g_size: usize) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let dli = (li as usize) + 1;
    let dlj = (lj as usize) + (lk as usize) + 1;

    let dj = dli; // g_stride_j = dli
    let dk = dli * dlj; // g_stride_k

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);

    let gx = 0usize;
    let gy = g_size;
    let gz = 2 * g_size;

    let mut out = vec![0.0_f64; nci * ncj * nck];

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let vx = g[gx + ix as usize + jx as usize * dj + kx as usize * dk];
                let vy = g[gy + iy as usize + jy as usize * dj + ky as usize * dk];
                let vz = g[gz + iz as usize + jz as usize * dj + kz as usize * dk];
                out[(k_idx * ncj + j_idx) * nci + i_idx] += vx * vy * vz;
            }
        }
    }

    out
}

/// Write a component-leading 3-component Cartesian gradient buffer into `staging`,
/// applying the cart→sph transform per component when the representation is
/// spheric. `cart_grad` is `[3 * nci*ncj*nck]` (each component i-fastest, k-slowest,
/// the `contract_3c1e_grad` layout). Spinor is rejected by the caller (D-06).
fn write_3c1e_grad_staging<F: CintFloat>(
    cart_grad: &[f64],
    li: u8,
    lj: u8,
    lk: u8,
    representation: Representation,
    staging: &mut [F],
) {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let cart_block = nci * ncj * nck;

    match representation {
        Representation::Spheric => {
            let nsi = nsph(li);
            let nsj = nsph(lj);
            let nsk = nsph(lk);
            let sph_block = nsi * nsj * nsk;
            for comp in 0..3usize {
                let src = &cart_grad[comp * cart_block..(comp + 1) * cart_block];
                let sph = cart_to_sph_3c1e(src, li, lj, lk);
                let base = comp * sph_block;
                let copy_len = sph.len().min(staging.len().saturating_sub(base));
                for n in 0..copy_len {
                    staging[base + n] = F::from_f64_lossy(sph[n]);
                }
            }
        }
        _ => {
            // Cart (spinor never reaches here).
            let copy_len = cart_grad.len().min(staging.len());
            for n in 0..copy_len {
                staging[n] = F::from_f64_lossy(cart_grad[n]);
            }
        }
    }
}

/// Build `ExecutionStats` for a completed 3c1e gradient launch.
fn grad_stats<F: CintFloat>(plan: &ExecutionPlan<'_>, staging: &[F]) -> ExecutionStats {
    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;
    let staging_bytes = staging.len() * std::mem::size_of::<F>();
    ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0,
        fallback_reason: plan.workspace.fallback_reason,
    }
}

/// `int3c1e_ip1` — ∇ on bra i of the 3-center OVERLAP (DRV1-03, half 1).
///
/// Pure Phase-21-style reuse of the existing overlap base `fill_g_tensor_3c1e`:
/// build at `li+1` headroom (libcint ng `{1,0,0,...}`), apply the 1e nabla on i
/// (`nabla1i_3c1e`), contract 3 components (`contract_3c1e_grad`). NO Rys, so NO
/// nroots guard (RESEARCH Pitfall 4 — skip the guard for ip1 ONLY). Max-l = f.
/// Spinor → UnsupportedApi (D-06). Host-side (matching the plan-03 2c2e gradient
/// precedent); the per-primitive loop folds the contraction coefficients.
fn launch_center_3c1e_ip1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int3c1e_ip1 gradient".to_owned(),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // common_factor = SQRTPI * PI * fac_sp(li) * fac_sp(lj) * fac_sp(lk).
    let common_factor = SQRTPI
        * std::f64::consts::PI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
    let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

    // For nctr==1 the common case is a single (ci,cj,ck)=(0,0,0) triple. Loop over
    // contraction columns (folding the coefficient product) into one cart buffer.
    let mut cart_grad = vec![0.0_f64; 3 * block_len];

    for kp in 0..n_prim_k {
        let ak = shell_k.exponents[kp];
        for jp in 0..n_prim_j {
            let aj = shell_j.exponents[jp];
            for ip in 0..n_prim_i {
                let ai = shell_i.exponents[ip];
                let aijk = ai + aj + ak;
                let eijk =
                    (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
                if eijk > 60.0 {
                    continue;
                }
                let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());

                // Coefficient product for nctr==1; the common dispatched case.
                let weight = shell_i.coefficients[ip * n_ctr_i]
                    * shell_j.coefficients[jp * n_ctr_j]
                    * shell_k.coefficients[kp * n_ctr_k];
                let fac = common_factor * dijk * weight;

                // Build the OVERLAP base at li+1 headroom (ng {1,0,0,...}).
                let g = fill_g_tensor_3c1e(
                    fac, ai, aj, ak, ri, rj, rk, rirj, li as u32 + 1, lj as u32, lk as u32,
                );
                let g1 = nabla1i_3c1e(&g, li as u32, lj as u32, lk as u32, ai);
                let gout = contract_3c1e_grad(&g, &g1, li, lj, lk);
                for (dst, &src) in cart_grad.iter_mut().zip(gout.iter()) {
                    *dst += src;
                }
            }
        }
    }

    write_3c1e_grad_staging::<F>(&cart_grad, li, lj, lk, plan.representation, staging);
    Ok(grad_stats::<F>(plan, staging))
}

/// `int3c1e_iprinv` — ∇ on bra i of the 3-center rinv-COULOMB (DRV1-03, half 2).
///
/// The ONLY genuinely-new base kernel in clusters A & B (RESEARCH Pitfall 1): the
/// gout is byte-identical to ip1's, but the BASE is the Rys-driven nuclear g-tensor
/// `fill_g_tensor_3c1e_nuc` (`CINTg3c1e_nuc`), not the overlap base. Reuses
/// `rys_roots_host` and the already-plumbed PTR_RINV_ORIG origin (D-08). Fail-closed
/// at nroots>5 BEFORE any rys call (D-13; fff → nroots 6 → reject). Spinor → UnsupportedApi.
fn launch_center_3c1e_iprinv<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int3c1e_iprinv gradient".to_owned(),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;

    // nrys_roots = (li_ceil + lj + lk)/2 + 1 with the ip headroom li_ceil = li+1
    // (g3c1e.c:41). Fail-closed > 5 BEFORE any rys_roots_host call (D-13; fff → 6).
    let nroots = ((li as usize + 1) + lj as usize + lk as usize) / 2 + 1;
    if nroots > 5 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nroots}"),
        });
    }

    // rinv origin (already plumbed, D-08). The validator rejected None earlier;
    // fail typed here too (never read a garbage origin) — defensive.
    let cr = plan
        .operator_env_params
        .rinv_orig
        .ok_or(cintxRsError::InvalidEnvParam {
            param: "PTR_RINV_ORIG",
            reason: "int3c1e_iprinv kernel reached with no rinv origin".to_owned(),
        })?;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let common_factor = SQRTPI
        * std::f64::consts::PI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
    let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

    let mut cart_grad = vec![0.0_f64; 3 * block_len];

    for kp in 0..n_prim_k {
        let ak = shell_k.exponents[kp];
        for jp in 0..n_prim_j {
            let aj = shell_j.exponents[jp];
            for ip in 0..n_prim_i {
                let ai = shell_i.exponents[ip];
                let aijk = ai + aj + ak;
                let eijk =
                    (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
                if eijk > 60.0 {
                    continue;
                }

                let weight = shell_i.coefficients[ip * n_ctr_i]
                    * shell_j.coefficients[jp * n_ctr_j]
                    * shell_k.coefficients[kp * n_ctr_k];
                // dijk = common_factor * ci*cj*ck * exp(-eijk) / aijk  (cint3c1e.c:317).
                let dijk = common_factor * weight * f64::exp(-eijk) / aijk;

                // rijk = (ai*ri + aj*rj + ak*rk)/aijk.
                let rijk = [
                    (ai * ri[0] + aj * rj[0] + ak * rk[0]) / aijk,
                    (ai * ri[1] + aj * rj[1] + ak * rk[1]) / aijk,
                    (ai * ri[2] + aj * rj[2] + ak * rk[2]) / aijk,
                ];

                // x = aijk * dist^2(rijk, cr) * tau^2, tau = 1 (point rinv, no
                // RINV_ZETA — CINTnuc_mod returns 1). (cint3c1e.c:325-327)
                let d0 = rijk[0] - cr[0];
                let d1 = rijk[1] - cr[1];
                let d2 = rijk[2] - cr[2];
                let x = aijk * (d0 * d0 + d1 * d1 + d2 * d2);
                let (u, w) = rys_roots_host::<f64>(nroots, x);

                // Sum over Rys roots: t2 = u/(1+u) (tau=1), fac = dijk * w[root].
                for root in 0..nroots {
                    let t2 = u[root] / (1.0 + u[root]);
                    let fac = dijk * w[root];
                    let g = fill_g_tensor_3c1e_nuc(
                        fac, t2, ai, aj, ak, ri, rj, rk, rijk, cr, li as u32 + 1,
                        lj as u32, lk as u32,
                    );
                    let g1 = nabla1i_3c1e(&g, li as u32, lj as u32, lk as u32, ai);
                    let gout = contract_3c1e_grad(&g, &g1, li, lj, lk);
                    for (dst, &src) in cart_grad.iter_mut().zip(gout.iter()) {
                        *dst += src;
                    }
                }
            }
        }
    }

    write_3c1e_grad_staging::<F>(&cart_grad, li, lj, lk, plan.representation, staging);
    Ok(grad_stats::<F>(plan, staging))
}

/// Generic inner for the 3c1e launcher.
///
/// Dispatches the [`center_3c1e_kernel`] device kernel (at f64) on `plan`'s
/// resolved backend, reads back the Cartesian buffer, then applies the
/// representation transform with the output cast to `F` (see module precision
/// policy). The signature is unchanged from the pre-GPU version so existing
/// callers/tests are unaffected.
///
/// # Normalization chain (from libcint):
/// common_factor = sqrt(pi)*pi * fac_sp_i * fac_sp_j * fac_sp_k (g3c1e EnvVars)
/// fac = common_factor * exp(-eijk)/(aijk*sqrt(aijk)) * ci*cj*ck (per primitive)
fn launch_center_3c1e_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "3c1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c1e",
            detail: format!(
                "canonical_family mismatch for 3c1e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    // Require exactly 3 shells
    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c1e",
            detail: format!(
                "3c1e kernel requires exactly 3 shells, got {}",
                shells.len()
            ),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Phase 23 DRV1-03: operator dispatch. 3c1e has NO operator dispatch in the
    // scalar path; ADD it here, BEFORE the scalar fall-through.
    //   ip1    = ∇ on bra i of the 3-center OVERLAP (no Rys, existing base).
    //   iprinv = ∇ on bra i of the 3-center rinv-COULOMB (Rys-driven NEW base,
    //            reusing the plumbed PTR_RINV_ORIG env slot, D-08).
    match plan.descriptor.operator_name() {
        "ip1" => return launch_center_3c1e_ip1::<F>(plan, staging),
        "iprinv" => return launch_center_3c1e_iprinv::<F>(plan, staging),
        _ => {} // fall through to the existing scalar overlap path
    }

    // Atom coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // common_factor = sqrt(pi) * pi * fac_sp(li) * fac_sp(lj) * fac_sp(lk)
    let common_factor = SQRTPI
        * std::f64::consts::PI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk);

    // Output size in Cartesian / spherical.
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    // Accumulated Cartesian integral buffer (i fastest, k slowest)
    let mut cart_buf = vec![0.0_f64; nci * ncj * nck];

    let n_prim_k = shell_k.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_i = shell_i.nprim as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_i = shell_i.nctr as usize;

    // expcutoff: libcint default EXPCUTOFF = 60.
    let expcutoff = 60.0_f64;

    // Primitive exponent arrays (shared across contraction columns).
    let exps_i: Vec<f64> = (0..n_prim_i).map(|ip| shell_i.exponents[ip]).collect();
    let exps_j: Vec<f64> = (0..n_prim_j).map(|jp| shell_j.exponents[jp]).collect();
    let exps_k: Vec<f64> = (0..n_prim_k).map(|kp| shell_k.exponents[kp]).collect();

    // Dispatch onto the resolved backend's device client (compute in f64), once
    // per (ci, cj, ck) contraction triple with that triple's coefficient
    // columns. The kernel folds the per-primitive coefficient product. For the
    // common nctr==1 case this is a single launch.
    for ck in 0..n_ctr_k {
        let coeff_k: Vec<f64> = (0..n_prim_k)
            .map(|kp| shell_k.coefficients[kp * n_ctr_k + ck])
            .collect();
        for cj in 0..n_ctr_j {
            let coeff_j: Vec<f64> = (0..n_prim_j)
                .map(|jp| shell_j.coefficients[jp * n_ctr_j + cj])
                .collect();
            for ci in 0..n_ctr_i {
                let coeff_i: Vec<f64> = (0..n_prim_i)
                    .map(|ip| shell_i.coefficients[ip * n_ctr_i + ci])
                    .collect();

                let prim_buf: Vec<f64> = match backend {
                    #[cfg(feature = "cpu")]
                    ResolvedBackend::Cpu(client) => run_3c1e_device::<cubecl::cpu::CpuRuntime>(
                        client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
                        n_prim_k as u32, ri, rj, rk, common_factor, expcutoff, &exps_i, &exps_j,
                        &exps_k, &coeff_i, &coeff_j, &coeff_k,
                    ),
                    #[cfg(feature = "wgpu")]
                    ResolvedBackend::Wgpu(client, _) => run_3c1e_device::<cubecl_wgpu::WgpuRuntime>(
                        client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
                        n_prim_k as u32, ri, rj, rk, common_factor, expcutoff, &exps_i, &exps_j,
                        &exps_k, &coeff_i, &coeff_j, &coeff_k,
                    ),
                    #[cfg(feature = "cuda")]
                    ResolvedBackend::Cuda(client) => run_3c1e_device::<cubecl_cuda::CudaRuntime>(
                        client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
                        n_prim_k as u32, ri, rj, rk, common_factor, expcutoff, &exps_i, &exps_j,
                        &exps_k, &coeff_i, &coeff_j, &coeff_k,
                    ),
                    #[cfg(feature = "rocm")]
                    ResolvedBackend::Rocm(client) => run_3c1e_device::<cubecl_hip::HipRuntime>(
                        client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
                        n_prim_k as u32, ri, rj, rk, common_factor, expcutoff, &exps_i, &exps_j,
                        &exps_k, &coeff_i, &coeff_j, &coeff_k,
                    ),
                    #[cfg(feature = "metal")]
                    ResolvedBackend::Metal(client, _) => run_3c1e_device::<cubecl_wgpu::WgpuRuntime>(
                        client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
                        n_prim_k as u32, ri, rj, rk, common_factor, expcutoff, &exps_i, &exps_j,
                        &exps_k, &coeff_i, &coeff_j, &coeff_k,
                    ),
                };

                // For nctr==1 (the dispatched-shell common case) ck==cj==ci==0
                // and prim_buf is the full contracted Cartesian block.
                for (dst, &src) in cart_buf.iter_mut().zip(prim_buf.iter()) {
                    *dst += src;
                }
            }
        }
    }

    // Apply cart-to-sph transform or copy Cartesian to staging.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_3c1e(&cart_buf, li, lj, lk);
            let sph_size = nsi * nsj * nsk;
            let copy_len = staging.len().min(sph_size);
            for (dst, &src) in staging[..copy_len].iter_mut().zip(sph[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
        _ => {
            let copy_len = staging.len().min(cart_buf.len());
            for (dst, &src) in staging[..copy_len].iter_mut().zip(cart_buf[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
    }

    // WR-06: precision-aware nonzero sentinel.
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

/// Launch the 3c1e kernel for a contracted shell triple.
///
/// Outer precision dispatcher: keeps the registered `FamilyLaunchFn` signature
/// so the `as FamilyLaunchFn` cast in `kernels/mod.rs` compiles unchanged.
/// Internally matches on `plan.precision` and delegates to the generic inner
/// `launch_center_3c1e_typed::<F>`, reinterpreting staging via `bytemuck::cast_slice_mut`
/// for the F32 arm (A5 proven sound in Plan 01).
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
pub fn launch_center_3c1e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_center_3c1e_typed::<f64>(backend, plan, specialization, staging)
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
            launch_center_3c1e_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Device kernel cross-check: the CubeCL kernel (on CpuRuntime, f64) must
    // reproduce the host `fill_g_tensor_3c1e` + `contract_3c1e_ovlp` reference.
    // ─────────────────────────────────────────────────────────────────────────

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host reference: single-primitive single-contraction shell triple, the
    /// same way the device kernel does, using `fill_g_tensor_3c1e`.
    #[allow(clippy::too_many_arguments)]
    fn host_cart_3c1e(
        ai: f64,
        aj: f64,
        ak: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        li: u8,
        lj: u8,
        lk: u8,
        common_factor: f64,
    ) -> Vec<f64> {
        let dli = (li as usize) + 1;
        let dlj = (lj as usize) + (lk as usize) + 1;
        let dlk = (lk as usize) + 1;
        let vrr_nmax = (li as usize) + (lj as usize) + (lk as usize) + 1;
        let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
        let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
        let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
        let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
        let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

        let aijk = ai + aj + ak;
        let eijk = (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
        let mut out = vec![0.0_f64; ncart(li) * ncart(lj) * ncart(lk)];
        if eijk > 60.0 {
            return out;
        }
        let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
        let fac = common_factor * dijk;

        let g = fill_g_tensor_3c1e(
            fac, ai, aj, ak, ri, rj, rk, rirj, li as u32, lj as u32, lk as u32,
        );
        let prim = contract_3c1e_ovlp(&g, li, lj, lk, g_alloc);
        for (dst, &src) in out.iter_mut().zip(prim.iter()) {
            *dst += src;
        }
        out
    }

    fn assert_device_matches_host(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 1.2, 0.3];
        let rk = [0.4_f64, -0.5, 1.1];
        let common_factor =
            SQRTPI * std::f64::consts::PI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

        let host = host_cart_3c1e(ai, aj, ak, ri, rj, rk, li, lj, lk, common_factor);
        let dev = run_3c1e_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lj as u32,
            lk as u32,
            1,
            1,
            1,
            ri,
            rj,
            rk,
            common_factor,
            60.0,
            &[ai],
            &[aj],
            &[ak],
            &[1.0],
            &[1.0],
            &[1.0],
        );

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch for li={li} lj={lj} lk={lk}"
        );
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch li={li} lj={lj} lk={lk} idx={idx}: \
                 host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn test_device_kernel_matches_host_sss() {
        assert_device_matches_host(0, 0, 0, 1.0, 1.2, 0.8);
    }

    #[test]
    fn test_device_kernel_matches_host_ssp() {
        assert_device_matches_host(0, 0, 1, 0.9, 1.1, 0.7);
    }

    #[test]
    fn test_device_kernel_matches_host_pss() {
        assert_device_matches_host(1, 0, 0, 0.7, 1.3, 0.9);
    }

    #[test]
    fn test_device_kernel_matches_host_spp() {
        assert_device_matches_host(0, 1, 1, 0.8, 1.0, 0.6);
    }

    #[test]
    fn test_device_kernel_matches_host_psp() {
        // li>0 with k>0: exercises i-HRR and k-separation HRR together.
        assert_device_matches_host(1, 0, 1, 0.6, 0.9, 1.2);
    }

    /// Genericity evidence: the kernel compiles and runs for `F = f32` as well
    /// as `f64` ("cubecl kernel with generics float"). Launch an s-s-s triple at
    /// f32 on the CPU runtime and assert a finite, positive result.
    #[test]
    fn test_center_3c1e_kernel_generic_f32() {
        let client = cpu_client();
        let exps_i = [1.0_f32];
        let exps_j = [1.0_f32];
        let exps_k = [1.0_f32];
        let coeff_dummy = [1.0_f32];
        // s-s-s: dli=dlj=dlk=1, vrr_nmax=1 → g_alloc=1 → 3*1.
        let g_zero = [0.0_f32; 3];
        let out_zero = [0.0_f32; 1];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&exps_j));
        let exps_k_h = client.create_from_slice(f32::as_bytes(&exps_k));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&coeff_dummy));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&coeff_dummy));
        let coeff_k_h = client.create_from_slice(f32::as_bytes(&coeff_dummy));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let common_factor = (SQRTPI
            * std::f64::consts::PI
            * common_fac_sp(0)
            * common_fac_sp(0)
            * common_fac_sp(0)) as f32;

        center_3c1e_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0,
            0.0,
            0.0,
            1.2,
            0.3,
            0.4,
            -0.5,
            1.1,
            common_factor,
            60.0_f32,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 3c1e kernel result must be finite");
        assert!(out > 0.0, "s-s-s 3c1e f32 result should be positive: {out}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1a: launch_center_3c1e_typed::<f64> byte-identical to the outer
    // launch_center_3c1e at f64 (center_3c1e_parity).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c1e_parity_f64() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(15), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(15), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        let result_outer = launch_center_3c1e(&backend, &plan, &spec, &mut staging_outer);
        assert!(result_outer.is_ok(), "outer f64 should succeed: {:?}", result_outer);

        let result_typed = launch_center_3c1e_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(result_typed.is_ok(), "typed f64 should succeed: {:?}", result_typed);

        assert_eq!(staging_outer[0].to_bits(), staging_typed[0].to_bits(),
            "f64 outer and typed should be byte-identical: outer={} typed={}", staging_outer[0], staging_typed[0]);
        assert!(staging_outer[0].is_finite() && staging_outer[0].abs() > 1e-20,
            "3c1e s-s-s overlap should be finite and nonzero: {}", staging_outer[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1b: launch_center_3c1e F32 path runs without panic.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c1e_f32_smoke() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(15), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(15), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_center_3c1e(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "F32 3c1e should succeed without panic: {:?}", result);

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(staging_f32[0].is_finite(), "F32 3c1e result should be finite: {}", staging_f32[0]);
        assert!(staging_f32[0] > 0.0, "F32 3c1e result should be positive: {}", staging_f32[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pure-host G-tensor unit tests (retained from the pre-GPU version).
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_fill_g_tensor_3c1e_sss() {
        let fac = 1.0_f64;
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ak = 1.0_f64;
        let ri = [0.0_f64; 3];
        let rj = [0.0_f64; 3];
        let rk = [0.0_f64; 3];
        let rirj = [0.0_f64; 3];

        let g = fill_g_tensor_3c1e(fac, ai, aj, ak, ri, rj, rk, rirj, 0, 0, 0);

        assert!((g[0] - 1.0).abs() < 1e-15, "gx[0] should be 1.0");
        assert!((g[1] - 1.0).abs() < 1e-15, "gy[0] should be 1.0");
        assert!((g[2] - 1.0).abs() < 1e-15, "gz[0] should be 1.0");
    }

    #[test]
    fn test_contract_3c1e_ovlp_sss() {
        let fac = 2.5_f64;
        let g = vec![1.0_f64, 1.0, fac]; // [gx|gy|gz] each of size 1
        let out = contract_3c1e_ovlp(&g, 0, 0, 0, 1);
        assert_eq!(out.len(), 1);
        assert!((out[0] - fac).abs() < 1e-14, "s-s-s overlap should equal gz[0] = {}", fac);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 23 DRV1-03: int3c1e_ip1 host gradient correctness via finite
    // difference of the 3c1e OVERLAP w.r.t. moving center i, on a NON-SQUARE
    // (p × s × s) block. A square block could hide an axis/layout bug; p×s×s
    // is deliberately rectangular in i vs j/k.
    // ─────────────────────────────────────────────────────────────────────────

    /// Single-primitive overlap cart block (i fastest, k slowest) at center i = ri.
    fn ovlp_block(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64, ri: [f64; 3], rj: [f64; 3], rk: [f64; 3]) -> Vec<f64> {
        let common_factor = SQRTPI * std::f64::consts::PI
            * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);
        host_cart_3c1e(ai, aj, ak, ri, rj, rk, li, lj, lk, common_factor)
    }

    /// Single-primitive int3c1e_ip1 cart gradient (component-leading).
    fn ip1_block(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64, ri: [f64; 3], rj: [f64; 3], rk: [f64; 3]) -> Vec<f64> {
        let common_factor = SQRTPI * std::f64::consts::PI
            * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);
        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
        let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
        let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
        let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
        let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];
        let aijk = ai + aj + ak;
        let eijk = (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
        let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
        let fac = common_factor * dijk;
        let g = fill_g_tensor_3c1e(fac, ai, aj, ak, ri, rj, rk, rirj, li as u32 + 1, lj as u32, lk as u32);
        let g1 = nabla1i_3c1e(&g, li as u32, lj as u32, lk as u32, ai);
        contract_3c1e_grad(&g, &g1, li, lj, lk)
    }

    #[test]
    fn test_int3c1e_ip1_matches_finite_difference_pss() {
        // NON-SQUARE: p (li=1) × s (lj=0) × s (lk=0). ni=3, nj=1, nk=1.
        let (li, lj, lk) = (1u8, 0u8, 0u8);
        let (ai, aj, ak) = (0.9_f64, 1.1, 0.7);
        let ri = [0.10_f64, -0.20, 0.30];
        let rj = [0.40_f64, 1.20, -0.30];
        let rk = [-0.50_f64, 0.40, 0.90];

        let analytic = ip1_block(li, lj, lk, ai, aj, ak, ri, rj, rk);
        let nci = ncart(li);
        let block = nci * ncart(lj) * ncart(lk);
        assert_eq!(analytic.len(), 3 * block, "ip1 output must be 3*ni*nj*nk");

        // Finite difference of the overlap w.r.t. moving center i along each axis.
        // libcint's gout `g1 = nabla1i = ∂χ/∂r` (the raising/lowering form), so the
        // integral derivative w.r.t. the CENTER is `∂I/∂A_i = -∫(∂χ/∂r)… = -g1·…`.
        // Hence the analytic ip1 block equals `-(central difference)` — the same
        // sign convention vendor parity checks against libcint.
        let h = 1e-6;
        for axis in 0..3usize {
            let mut rip = ri;
            let mut rim = ri;
            rip[axis] += h;
            rim[axis] -= h;
            let op = ovlp_block(li, lj, lk, ai, aj, ak, rip, rj, rk);
            let om = ovlp_block(li, lj, lk, ai, aj, ak, rim, rj, rk);
            for n in 0..block {
                let fd = -(op[n] - om[n]) / (2.0 * h);
                let an = analytic[axis * block + n];
                let diff = (fd - an).abs();
                let thr = 1e-6 + 1e-5 * an.abs();
                assert!(
                    diff <= thr,
                    "int3c1e_ip1 axis={axis} n={n}: analytic={an:.12e} fd={fd:.12e} diff={diff:.3e}"
                );
            }
        }
    }
}
