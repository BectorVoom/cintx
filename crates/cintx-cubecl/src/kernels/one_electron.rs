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
use crate::math::obara_saika::{hrr_step_host, vrr_step_host};
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_2d;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

/// sqrt(pi) constant — used in G-tensor base case normalization.
/// Matches libcint `g1e.c` `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

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

/// Contract G-tensor elements for the overlap operator.
///
/// Loops over all (ix+jx, iy+jy, iz+jz) Cartesian products and returns the
/// flat cartesian integral buffer of size ncart(li) * ncart(lj).
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
            let g3x = 4.0 * aj * aj * g[gx + nx + 2 * dj]
                - 2.0 * aj * (2.0 * jxf + 1.0) * vx0
                + jxf * (jxf - 1.0) * if jx >= 2 { g[gx + nx - 2 * dj] } else { 0.0 };

            let jyf = jy as f64;
            let g3y = 4.0 * aj * aj * g[gy + ny + 2 * dj]
                - 2.0 * aj * (2.0 * jyf + 1.0) * vy0
                + jyf * (jyf - 1.0) * if jy >= 2 { g[gy + ny - 2 * dj] } else { 0.0 };

            let jzf = jz as f64;
            let g3z = 4.0 * aj * aj * g[gz + nz + 2 * dj]
                - 2.0 * aj * (2.0 * jzf + 1.0) * vz0
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

/// Compute nuclear attraction integrals for one primitive pair, all atoms.
///
/// Uses Rys quadrature with Boys-weighted VRR.
/// Reference: g1e.c lines 208-320 (CINTg1e_nuc).
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
        let x_boys =
            pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

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

    // Suppress backend: host-side pipeline executes natively without GPU dispatch.
    let _ = backend;

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

    if !is_overlap && !is_kinetic && !is_nuclear {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("1e operator '{}' is not supported", op_name),
        });
    }

    // Output sizes
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    // Primitive loop over (pi, pj) pairs
    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    // Per-contraction-pair Cartesian accumulators. Generally-contracted shells
    // (nctr>1, e.g. ANO/ANO-RCC) emit ONE nci*ncj Cartesian block per (ci,cj)
    // contraction pair. The contraction index is the OUTER (slow) AO index within
    // each shell axis and the angular component the inner index — libcint
    // "contraction-major" AO ordering (matches PySCF sph_labels: 2px,2py,2pz,
    // 3px,3py,3pz,...). For the segmented case (n_ctr_i == n_ctr_j == 1) this is
    // a single block and the result is byte-identical to the prior single-buffer
    // path. Block (ci,cj) starts at flat offset (ci*n_ctr_j + cj) * block_len.
    let block_len = nci * ncj;
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * block_len];

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        // norm_i: normalization factor; use 1.0 per-primitive (coefficients carry norms)
        let norm_i = 1.0_f64;

        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let norm_j = 1.0_f64;

            // Pair data
            let pd = compute_pdata_host(
                ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], norm_i, norm_j,
            );

            // Compute integral for this primitive pair
            let prim_buf = if is_overlap {
                let nmax = (li + lj) as u32;
                let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32);
                contract_overlap(&g, li, lj, nmax)
            } else if is_kinetic {
                // Kinetic requires two extra j-levels (lj+2) so D_j^2 can access g0[jx+2].
                // HRR to lj+2 levels requires nmax = li + lj + 2 VRR bra levels so
                // there are enough starting points for the two extra HRR steps.
                let nmax = (li + lj) as u32 + 2;
                let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32 + 2);
                contract_kinetic(&g, li, lj, nmax, aj)
            } else {
                // Nuclear attraction
                contract_nuclear(&pd, ri, rj, li, lj, atoms)
            };

            // Scatter this primitive pair into every contraction block, weighted by
            // that contraction's coefficient pair. Shell coefficient layout is
            // coefficients[pi * n_ctr + ctr_idx] (libcint convention). Each (ci,cj)
            // block accumulates independently — the previous code summed every
            // (ci,cj) pair into a single buffer, truncating nctr>1 shells.
            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for cj in 0..n_ctr_j {
                    let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                    let weight = coeff_i * coeff_j;
                    let base = (ci * n_ctr_j + cj) * block_len;
                    for k in 0..block_len {
                        cart_blocks[base + k] += weight * prim_buf[k];
                    }
                }
            }
        }
    }

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
            // cart_blocks is exactly one nci*ncj block here — identical to the old
            // cart_buf. Matches libcint c2s_sf_1e for int1e_*_spinor.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            cart_to_spinor_sf_2d::<F>(staging, &cart_blocks, li, kappa_i, lj, kappa_j)?;
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
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
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
            launch_one_electron_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
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
        let pd_disp = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

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
        assert!(t_ss > 0.0, "s-s kinetic integral should be positive, got {t_ss}");
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
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

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
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(0), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(0), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 1];

        // Call the generic _typed inner directly (f64 monomorphization).
        // RED: compile fails until launch_one_electron_typed is defined.
        let result = launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "f64 inner should succeed: {:?}", result);
        assert!(staging[0].is_finite(), "f64 overlap should be finite, got {}", staging[0]);
        assert!(staging[0] > 0.0, "s-s overlap should be positive, got {}", staging[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-1b: launch_one_electron_typed::<f32> writes a positive finite
    // f32 s-s overlap. RED: compile fails until the typed inner exists; then
    // fails at runtime until f32 math is correctly wired.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_precision_dispatch_f32_inner_positive_overlap() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

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
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(0), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(0), Representation::Cart, &basis, shells, &query).unwrap();
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
        assert!(staging_f32[0].is_finite(), "f32 overlap should be finite, got {}", staging_f32[0]);
        assert!(staging_f32[0] > 0.0, "s-s overlap (f32) should be positive, got {}", staging_f32[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: rys_root2_host returns valid roots (0,1) and positive weights
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_rys_root2_host_valid_roots() {
        for x in [0.01, 0.5, 2.0, 5.0, 15.0, 35.0, 45.0] {
            let (u, w) = rys_root2_host(x);
            assert!(u[0] >= 0.0, "root u[0] should be non-negative for x={x}, got {}", u[0]);
            assert!(u[1] >= 0.0, "root u[1] should be non-negative for x={x}, got {}", u[1]);
            assert!(w[0] > 0.0, "weight w[0] should be positive for x={x}, got {}", w[0]);
            assert!(w[1] > 0.0, "weight w[1] should be positive for x={x}, got {}", w[1]);
            assert!(u[0] <= u[1], "roots should be ordered u[0] <= u[1] for x={x}");
        }
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
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

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
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
            let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
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

        let gc = Arc::new(Shell::try_new(0, 0, 2, 2, 0, Representation::Spheric, exps.clone(), coeffs_gc).unwrap());
        let c0 = Arc::new(Shell::try_new(0, 0, 2, 1, 0, Representation::Spheric, exps.clone(), coeffs_c0).unwrap());
        let c1 = Arc::new(Shell::try_new(0, 0, 2, 1, 0, Representation::Spheric, exps, coeffs_c1).unwrap());

        let block = overlap(gc.clone(), gc.clone());
        assert_eq!(block.len(), 4, "gc s nctr=2 self-overlap must be a 2x2 block, got {}", block.len());

        let s00 = overlap(c0.clone(), c0.clone())[0];
        let s01 = overlap(c0.clone(), c1.clone())[0];
        let s11 = overlap(c1.clone(), c1.clone())[0];

        // Contraction-major, bra-fastest (di_sph = 2, nsi = 1): block[ci + cj*2].
        assert!((block[0] - s00).abs() < 1e-12, "S[0,0] {} vs {}", block[0], s00);
        assert!((block[2] - s01).abs() < 1e-12, "S[0,1] {} vs {}", block[2], s01);
        assert!((block[1] - s01).abs() < 1e-12, "S[1,0] {} vs {}", block[1], s01);
        assert!((block[3] - s11).abs() < 1e-12, "S[1,1] {} vs {}", block[3], s11);
        assert!(block[0] > 0.0 && block[3] > 0.0, "diagonal must be positive (PSD)");
        assert!((block[1] - block[2]).abs() < 1e-12, "block must be symmetric");
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
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

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
            let all: Arc<[Arc<Shell>]> = Arc::from(vec![bra.clone(), ket.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
            let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
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

        let p_gc = Arc::new(Shell::try_new(0, 1, 2, 2, 0, Representation::Spheric, p_exps.clone(), p_gc_co).unwrap());
        let p_c0 = Arc::new(Shell::try_new(0, 1, 2, 1, 0, Representation::Spheric, p_exps.clone(), p_c0_co).unwrap());
        let p_c1 = Arc::new(Shell::try_new(0, 1, 2, 1, 0, Representation::Spheric, p_exps, p_c1_co).unwrap());
        let s_ket = Arc::new(Shell::try_new(
            1, 0, 1, 1, 0, Representation::Spheric,
            Arc::from(vec![0.9_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice()),
        ).unwrap());

        let block = overlap(p_gc, s_ket.clone()); // 6x1: di_sph=2*3=6, dj=1
        assert_eq!(block.len(), 6, "gc p nctr=2 vs s must be a 6x1 block, got {}", block.len());

        let seg0 = overlap(p_c0, s_ket.clone()); // 3 components of contraction 0
        let seg1 = overlap(p_c1, s_ket);         // 3 components of contraction 1
        assert_eq!(seg0.len(), 3);
        assert_eq!(seg1.len(), 3);

        // Contraction-major: rows [0..3) = contraction 0, rows [3..6) = contraction 1.
        for m in 0..3 {
            assert!((block[m] - seg0[m]).abs() < 1e-12, "ctr0 comp {m}: {} vs {}", block[m], seg0[m]);
            assert!((block[3 + m] - seg1[m]).abs() < 1e-12, "ctr1 comp {m}: {} vs {}", block[3 + m], seg1[m]);
        }
        // The gc block must NOT be truncated: contraction-1 rows are non-zero.
        assert!(block[3..6].iter().any(|v| v.abs() > 1e-12), "contraction-1 rows must be populated, not truncated to zero");
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
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

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
            let s_la = Arc::new(Shell::try_new(
                0, la, 1, 1, 0, Representation::Spheric,
                Arc::from(vec![0.9_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            ).unwrap());
            let s_lb = Arc::new(Shell::try_new(
                1, lb, 1, 1, 0, Representation::Spheric,
                Arc::from(vec![0.6_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            ).unwrap());
            let all: Arc<[Arc<Shell>]> = Arc::from(vec![s_la.clone(), s_lb.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let (bra, ket) = if swapped { (s_lb.clone(), s_la.clone()) } else { (s_la.clone(), s_lb.clone()) };
            let n = bra.ao_per_shell() * ket.ao_per_shell();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
            let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
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
                assert!(ab.iter().any(|v| v.abs() > 1e-10), "{op_sym} l=({la},{lb}) block unexpectedly all-zero");
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
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

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
            let d_gc = Arc::new(Shell::try_new(0, 2, 2, 2, 0, Representation::Spheric, d_exps.clone(), d_co.clone()).unwrap());
            let f_gc = Arc::new(Shell::try_new(1, 3, 2, 2, 0, Representation::Spheric, f_exps.clone(), f_co.clone()).unwrap());
            let all: Arc<[Arc<Shell>]> = Arc::from(vec![d_gc.clone(), f_gc.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let (bra, ket) = if swapped { (f_gc.clone(), d_gc.clone()) } else { (d_gc.clone(), f_gc.clone()) };
            let n = bra.ao_per_shell() * ket.ao_per_shell();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
            let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
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
        assert!(ab.iter().any(|v| v.abs() > 1e-10), "d_gc-f_gc block unexpectedly all-zero");
    }
}
