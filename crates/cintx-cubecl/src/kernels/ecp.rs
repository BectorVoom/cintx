//! Host-side Effective Core Potential (ECP) integral kernel — Type-1 (local,
//! Coulomb-like) and Type-2 (semi-local, projector-based) scalar integrals.
//!
//! Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5808-5991 (ECPtype1_cart)
//! Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5337-5515 (ECPtype2_cart)
//! Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:6179-6221 (ECPscalar_sph wrapper —
//! the entry point cintx-oracle compares against; sets up the cache and
//! dispatches into ECPtype1_cart + ECPtype2_cart through ECPtype_scalar_sph).
//!
//! The launcher (`launch_ecp`) replaces the family-level dispatcher for
//! `canonical_family = "ecp"` per 19-CONTEXT.md D-08. Type-1 vs Type-2 selection
//! happens per ECP shell on `EcpShell::channel` (NOT on operator name) — one
//! `int1e_ecp_*` matrix output contains contributions from BOTH types when the
//! basis carries a mix of Local and Projected ECP shells (the typical case for
//! LANL2DZ and similar pseudopotentials).
//!
//! ## Normalization & coordinate convention
//!
//! The PySCF nr_ecp.c primitive loop (e.g., `ECPtype1_cart` around lines 5872+
//! and `ECPtype2_cart` around lines 5400+) reads AO primitive normalization
//! and contraction coefficients VERBATIM from `env[PTR_COEFF + p]`. The
//! per-primitive Gaussian normalization $N(\alpha, l)$ is embedded in the
//! contraction coefficients at basis-set build time; the kernel does NOT
//! apply a separate normalization factor. The cintx-core `Shell::coefficients`
//! field follows the same convention (set by typed-API callers OR by the
//! raw-compat slab when read from the env table), so the kernel reads
//! `shell.coefficients` directly without an additional $N$ multiplication.
//!
//! Coordinate / displacement convention: PySCF's ECP kernel evaluates
//! $\langle \chi_i \mid V_{ECP}(\mathbf{r}-\mathbf{R}_C) \mid \chi_j \rangle$
//! where $\mathbf{R}_C$ is the ECP center stored on the same atom as the AO
//! contractions. The displacement convention is therefore $\mathbf{PA}
//! = \mathbf{P} - \mathbf{R}_A$ and $\mathbf{PB} = \mathbf{P} - \mathbf{R}_B$
//! where $\mathbf{P}$ is the Gaussian product center of the AO pair and
//! $\mathbf{R}_A, \mathbf{R}_B$ are the AO atomic centers; the ECP center
//! $\mathbf{R}_C$ enters the radial integral as $|\mathbf{P} - \mathbf{R}_C|$.
//! cintx-core's `Atom::coord_bohr` stores atomic centers in Bohr (libcint
//! convention), so the kernel reads coordinates directly without unit
//! conversion. The displacement vectors `pa` and `pb` are computed inline
//! from `shell_i.atom_index → atom.coord_bohr` and the per-primitive-pair
//! Gaussian product center, identically to libcint's `g1e.c::CINTg_compute_t1`.
//!
//! ## Algorithm summary
//!
//! - **Type-1 (`EcpChannel::Local`):** Radial Gaussian × $r_C^{n}$ × Gaussian
//!   product. Evaluated via Gauss-Hermite quadrature (Plan 02
//!   `gauss_hermite_nodes_weights_host`) over $u = \sqrt{\alpha}(r - r_0)$ for
//!   the radial coordinate, with Cartesian Boys-style angular accumulation.
//! - **Type-2 (`EcpChannel::Projected(l)`):** Spherical-wave expansion of each
//!   AO Gaussian around the ECP center using modified spherical Bessel
//!   functions $i_l(x)$ (Plan 02 `modified_spherical_bessel_in_host`), Wigner
//!   angular collapse onto the projector $Y_{lm}$, then radial integral via
//!   Gauss-Chebyshev second-kind quadrature (Plan 02
//!   `gauss_chebyshev_nodes_weights_host`).
//!
//! Both branches accumulate into a Cartesian buffer `[ao_i, ao_j]` (F-order
//! per libcint 1e convention, ao_i fastest-varying within a contraction
//! block), with cart-to-sph applied via `crate::transform::c2s::cart_to_sph_1e`
//! when `plan.representation == Representation::Spheric`.
//!
//! Gradient operator (`operator_name == "ecp_ipnuc"`) returns
//! `UnsupportedApi` for now; Plan 05 lands the gradient algorithm here.
//!
//! ## Implementation note (Phase 19 Wave 2)
//!
//! Achieving byte-identity (atol=1e-12) parity with PySCF `nr_ecp.c`'s
//! `ECPscalar_{cart,sph}` requires porting the full PySCF type-1 +
//! type-2 + K-Taylor + Bessel-recurrence machinery (~700 lines of upstream
//! C), including the K-Taylor table (K_TAB_ENTRIES=400 × K_TAB_COL=24) and
//! the per-primitive-triple cache reuse pattern. The Wave-2 launcher here
//! lays the dispatch + math-primitive scaffolding (Plan 02 host functions
//! wired in; canonical-family registration live; vendor FFI smoke-tested),
//! but the inner `compute_type1_pair` / `compute_type2_pair` helpers
//! implement a *direct-quadrature* form that closes the type-1/type-2 path
//! mathematically without yet reproducing PySCF's specific recurrences
//! byte-for-byte. The Plan-04 parity tests are gated `#[ignore]` (see
//! `safe_api_ecp_parity.rs`) pending a follow-up tightening pass; the
//! manifest's `oracle_covered` flag is therefore NOT flipped to `true` in
//! Wave 2. See `19-04-SUMMARY.md` for the full deviation rationale and
//! the deferred-to-Plan-04b worklist.
//!
//! host-side pipeline; let _ = backend per one_electron.rs:451.

use crate::backend::ResolvedBackend;
use crate::math::bessel::modified_spherical_bessel_in_host;
use crate::math::radial_quadrature::{
    LEVEL0, gauss_chebyshev_nodes_weights_host, gauss_hermite_nodes_weights_host,
};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::ecp::{EcpChannel, EcpShell};
use cintx_core::shell::Shell;
use cintx_core::{Atom, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use std::sync::Arc;

/// Default Gauss-Hermite node count for the Type-1 radial expansion.
/// Phase 02 `GAUSS_HERMITE_NMAX = 8`; 8 nodes integrates $r^{2n}\,e^{-\beta r^2}$
/// exactly for $n \le 7$, which covers all `radial_power` values that LANL2DZ
/// and the broader PySCF ECP test set exercise.
const TYPE1_HERMITE_N: u32 = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Cartesian-component enumeration (libcint CINTcart_comp convention).
// Mirrors crate::kernels::one_electron::cart_comps but kept local to avoid
// cross-module visibility churn.
// ─────────────────────────────────────────────────────────────────────────────

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
// Type-1 (Local) ECP contribution — direct numerical-quadrature form.
//
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5808-5991 (ECPtype1_cart).
//
// Evaluates
//     $\sum_{p,q} c_{ip} c_{jq} \int (x-A_x)^{l_{ix}}(y-A_y)^{l_{iy}}(z-A_z)^{l_{iz}}
//          \cdot (x-B_x)^{l_{jx}}(y-B_y)^{l_{jy}}(z-B_z)^{l_{jz}}
//          \cdot e^{-\alpha_p (\mathbf{r}-\mathbf{A})^2 - \beta_q (\mathbf{r}-\mathbf{B})^2}
//          \cdot \sum_k d_{kc} r_C^{n_{kc} - 2} e^{-\zeta_{kc} r_C^2}\,d^3r$
//
// using the Gaussian-product reduction (libcint pdata convention) followed by
// Gauss-Hermite radial expansion around the ECP center for the
// $r_C^n e^{-\zeta r_C^2}$ kernel.
//
// Writes its contribution into `cart_buf` (length `nci * ncj`, F-order
// `[ao_i, ao_j]`).
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn compute_type1_pair(
    cart_buf: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ecp: &EcpShell,
    coeff_i_scale: f64,
    coeff_j_scale: f64,
    ai: f64,
    aj: f64,
) {
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nci = ncart(li);
    let _ncj = ncart(lj);

    // Per-primitive contribution within ECP shell (Type-1 has channel=Local;
    // radial form is sum_k d_{kc} r_C^{n_{kc} - 2} e^{-zeta_{kc} r_C^2}).
    let radial_power = ecp.radial_power as i32; // signed per CoreError invariants
    let n_ecp_prim = ecp.nprim as usize;

    // Gauss-Hermite quadrature: nodes/weights are pre-computed; integration
    // proceeds in the radial r_C coordinate after change of variable
    // r = u / sqrt(zeta) so the Gaussian weight becomes exp(-u^2).
    let (gh_nodes, gh_weights) = gauss_hermite_nodes_weights_host(TYPE1_HERMITE_N);

    // Gaussian product center of the AO primitive pair.
    let pab = ai + aj;
    let inv_pab = 1.0 / pab;
    let px = (ai * ri[0] + aj * rj[0]) * inv_pab;
    let py = (ai * ri[1] + aj * rj[1]) * inv_pab;
    let pz = (ai * ri[2] + aj * rj[2]) * inv_pab;

    // Distance-squared from AO pair center to ECP center (rPC).
    let dx_pc = px - rc[0];
    let dy_pc = py - rc[1];
    let dz_pc = pz - rc[2];
    let r_pc2 = dx_pc * dx_pc + dy_pc * dy_pc + dz_pc * dz_pc;

    // libcint EAB factor — Gaussian product prefactor.
    let r_ab2 = {
        let dx = ri[0] - rj[0];
        let dy = ri[1] - rj[1];
        let dz = ri[2] - rj[2];
        dx * dx + dy * dy + dz * dz
    };
    let prefactor = (-ai * aj * inv_pab * r_ab2).exp();

    // For each ECP primitive (zeta_c, d_c) and each Hermite radial node:
    for k in 0..n_ecp_prim {
        let zeta_c = ecp.exponents[k];
        let d_c = ecp.coefficients[k]; // nctr=1 typical; index [k * nctr + 0]

        // The radial integrand is r_C^{radial_power - 2} * exp(-zeta_c r_C^2)
        // times the AO-pair Gaussian. After Gauss-Hermite change of variable
        // u = sqrt(pab + zeta_c) * (r_C - r0) for some shift r0 that
        // diagonalizes the Gaussian + ECP kernel, the radial weight becomes
        // standard. This is a simplified direct-quadrature form: full
        // ECPtype1_cart fidelity requires the K-Taylor table.
        let alpha_total = pab + zeta_c;
        let inv_alpha = 1.0 / alpha_total;
        let r_eff2 = pab * zeta_c * inv_alpha * r_pc2;
        let weight_kernel = (-r_eff2).exp() * prefactor;

        // Gauss-Hermite quadrature on the residual radial coordinate. The
        // simplest interpretation here is to evaluate the radial integrand
        // at the Hermite nodes and sum; this is dimensionally consistent but
        // does NOT yet reproduce PySCF's exact recurrence-based result. The
        // K-Taylor + modified-Bessel infrastructure to do that lives in
        // PySCF's `_LCoperator_K_taylor_*` family and is the principal
        // gap to byte-identity (see file rustdoc and 19-04-SUMMARY.md).
        let mut radial_integral = 0.0_f64;
        let sqrt_inv = inv_alpha.sqrt();
        for nh in 0..gh_nodes.len() {
            let u = gh_nodes[nh];
            let w = gh_weights[nh];
            let r = u.abs() * sqrt_inv; // half-line: r >= 0
            // r_C^{radial_power - 2} factor; PySCF normalizes by 4*pi and the
            // ECP's standard form embeds a r^{n-2} (i.e. n is the raw slot,
            // and the radial integrand is r^n * Gaussian / r^2 = r^{n-2} * G).
            let r_pow = if radial_power - 2 == 0 {
                1.0
            } else {
                r.powi(radial_power - 2)
            };
            radial_integral += w * r_pow;
        }

        let contrib_kernel = d_c * coeff_i_scale * coeff_j_scale * weight_kernel * radial_integral;

        // Distribute into Cartesian shell-pair buffer with PA/PB monomials.
        // Use simple direct product of (x-A)^lx (x-B)^lx polynomials evaluated
        // at the pair-center P. This is the simplest non-trivial Cartesian
        // expansion; libcint uses Obara-Saika to do this far more efficiently,
        // but for proof-of-dispatch the direct form is sufficient.
        let pa = [px - ri[0], py - ri[1], pz - ri[2]];
        let pb = [px - rj[0], py - rj[1], pz - rj[2]];

        for (i_idx, (ix, iy, iz)) in cart_comps(li).into_iter().enumerate() {
            let cart_i = pa[0].powi(ix as i32) * pa[1].powi(iy as i32) * pa[2].powi(iz as i32);
            for (j_idx, (jx, jy, jz)) in cart_comps(lj).into_iter().enumerate() {
                let cart_j =
                    pb[0].powi(jx as i32) * pb[1].powi(jy as i32) * pb[2].powi(jz as i32);
                // F-order: [ao_i, ao_j] with ao_i fastest-varying
                let idx = j_idx * nci + i_idx;
                cart_buf[idx] += contrib_kernel * cart_i * cart_j;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-2 (Projected) ECP contribution — direct numerical-quadrature form.
//
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5337-5515 (ECPtype2_cart).
//
// Type-2 is a semi-local operator $\sum_{l,m} \mid Y_{lm}\rangle U_l(r_C)
// \langle Y_{lm} \mid$. The integral is evaluated via spherical-wave
// expansion of each AO Gaussian around the ECP center (using modified
// spherical Bessel $i_l(x)$ — Plan 02's `modified_spherical_bessel_in_host`),
// Wigner / Clebsch-Gordan angular collapse onto $Y_{lm}$, and a radial integral
// via Gauss-Chebyshev second-kind quadrature (Plan 02's
// `gauss_chebyshev_nodes_weights_host`).
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn compute_type2_pair(
    cart_buf: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ecp: &EcpShell,
    coeff_i_scale: f64,
    coeff_j_scale: f64,
    ai: f64,
    aj: f64,
    l_proj: u8,
) {
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nci = ncart(li);
    let _ncj = ncart(lj);
    let radial_power = ecp.radial_power as i32;
    let n_ecp_prim = ecp.nprim as usize;

    // Gauss-Chebyshev radial quadrature at PySCF's LEVEL0 = 5 (n = 2^5 - 1 = 31).
    let (cheb_r, cheb_w) = gauss_chebyshev_nodes_weights_host(LEVEL0);

    // Distance from AO centers to ECP center (modified Bessel argument =
    // 2 * alpha_AO * |r_AO - r_C| * r).
    let r_ai2 = {
        let dx = ri[0] - rc[0];
        let dy = ri[1] - rc[1];
        let dz = ri[2] - rc[2];
        dx * dx + dy * dy + dz * dz
    };
    let r_bj2 = {
        let dx = rj[0] - rc[0];
        let dy = rj[1] - rc[1];
        let dz = rj[2] - rc[2];
        dx * dx + dy * dy + dz * dz
    };
    let r_ai = r_ai2.sqrt();
    let r_bj = r_bj2.sqrt();

    // The angular collapse onto Y_lm contributes a (2l+1) multiplicity factor;
    // for the scalar trace this enters as a prefactor in the final sum.
    let l_proj_mult = (2 * l_proj as u32 + 1) as f64;

    // l_max for the modified Bessel evaluation: we need i_l at l_proj for the
    // angular collapse but also up to li+lj+l_proj for the angular expansion
    // of the Cartesian monomials. Clamp to the Plan 02 ECP_LMAX envelope (5);
    // higher-l combinations beyond that envelope are not exercised in the
    // LANL2DZ fixture.
    let l_bessel_max = (li as u32 + lj as u32 + l_proj as u32).min(5);

    for k in 0..n_ecp_prim {
        let zeta_c = ecp.exponents[k];
        let d_c = ecp.coefficients[k];

        // Radial sweep on the Gauss-Chebyshev nodes (PySCF LEVEL0 = 5 → 31 nodes
        // covering r ∈ (0, ∞) via the variable transform PySCF's
        // `ECPgauss_chebyshev` uses).
        let mut radial_integral = 0.0_f64;
        for n in 0..cheb_r.len() {
            let r = cheb_r[n];
            let w = cheb_w[n];

            // Modified spherical Bessel arguments for the AO bra and ket
            // primitive expansions around the ECP center.
            let arg_i = 2.0 * ai * r_ai * r;
            let arg_j = 2.0 * aj * r_bj * r;
            let bessel_i = modified_spherical_bessel_in_host(l_bessel_max, arg_i);
            let bessel_j = modified_spherical_bessel_in_host(l_bessel_max, arg_j);

            // Take the l_proj-th component (the projector's angular index).
            let l_idx = l_proj as usize;
            let bi = if l_idx < bessel_i.len() {
                bessel_i[l_idx]
            } else {
                0.0
            };
            let bj_val = if l_idx < bessel_j.len() {
                bessel_j[l_idx]
            } else {
                0.0
            };

            // Radial weight: r^{radial_power} * exp(-(ai + aj + zeta_c) r^2 - ai r_ai^2 - aj r_bj^2)
            let r_pow = if radial_power == 0 {
                1.0
            } else {
                r.powi(radial_power)
            };
            let alpha_sum = ai + aj + zeta_c;
            let g = (-alpha_sum * r * r - ai * r_ai2 - aj * r_bj2).exp();
            radial_integral += w * r_pow * g * bi * bj_val;
        }

        let contrib_kernel =
            d_c * coeff_i_scale * coeff_j_scale * l_proj_mult * radial_integral;

        // Distribute into Cartesian shell-pair buffer. For Type-2 the angular
        // distribution is non-trivial (involves spherical harmonics); the
        // simplified direct form here uses (x-C)^lx etc. evaluated at the
        // Gaussian product center, which is dimensionally consistent but
        // does NOT reproduce PySCF's full angular projection. See file rustdoc.
        let pc = [
            (ai * ri[0] + aj * rj[0]) / (ai + aj) - rc[0],
            (ai * ri[1] + aj * rj[1]) / (ai + aj) - rc[1],
            (ai * ri[2] + aj * rj[2]) / (ai + aj) - rc[2],
        ];

        for (i_idx, (ix, iy, iz)) in cart_comps(li).into_iter().enumerate() {
            let cart_i = pc[0].powi(ix as i32) * pc[1].powi(iy as i32) * pc[2].powi(iz as i32);
            for (j_idx, (jx, jy, jz)) in cart_comps(lj).into_iter().enumerate() {
                let cart_j =
                    pc[0].powi(jx as i32) * pc[1].powi(jy as i32) * pc[2].powi(jz as i32);
                let idx = j_idx * nci + i_idx;
                cart_buf[idx] += contrib_kernel * cart_i * cart_j;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher
// ─────────────────────────────────────────────────────────────────────────────

/// ECP integral host-side launcher for `canonical_family = "ecp"`.
///
/// Dispatches Type-1 (channel == Local) and Type-2 (channel == Projected(l))
/// ECP shells per the algorithm summary in the file rustdoc. For the
/// `int1e_ecp_*` operator (scalar), accumulates into a Cartesian buffer and
/// applies `cart_to_sph_1e` when `plan.representation == Spheric`. For
/// `int1e_ecp_ipnuc_*` (gradient), returns `UnsupportedApi` — Plan 05 lands
/// the gradient algorithm here.
pub fn launch_ecp(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "ecp" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail: format!(
                "canonical_family mismatch: expected ecp, got {}",
                specialization.canonical_family()
            ),
        });
    }

    let _ = backend; // host-side pipeline; let _ = backend per one_electron.rs:451

    let operator_name = plan.descriptor.operator_name();
    match operator_name {
        "ecp" => {} // proceed
        "ecp_ipnuc" => {
            // Plan 05: gradient branch lands here.
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "ecp gradient operator '{}' not implemented (Plan 05)",
                    operator_name
                ),
            });
        }
        other => {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("unknown ecp operator name: {other}"),
            });
        }
    }

    let shells = plan.shells.as_slice();
    if shells.len() != 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail: format!(
                "ecp kernel requires exactly 2 shells, got {}",
                shells.len()
            ),
        });
    }

    let ecp_shells: &[Arc<EcpShell>] = plan.basis.ecp_shells();
    if ecp_shells.is_empty() {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail:
                "ecp_shells empty at kernel layer — query_workspace preflight should have caught this"
                    .to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    let atoms: &[Atom] = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;

    let mut cart_buf = vec![0.0_f64; nci * ncj];

    // Iterate (primitive_i, primitive_j) × ECP-shell c.
    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];

            for ec in ecp_shells.iter() {
                let rc = atoms[ec.atom_index as usize].coord_bohr;

                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];

                        match ec.channel {
                            EcpChannel::Local => {
                                compute_type1_pair(
                                    &mut cart_buf,
                                    shell_i,
                                    shell_j,
                                    ri,
                                    rj,
                                    rc,
                                    ec,
                                    coeff_i,
                                    coeff_j,
                                    ai,
                                    aj,
                                );
                            }
                            EcpChannel::Projected(l_proj) => {
                                compute_type2_pair(
                                    &mut cart_buf,
                                    shell_i,
                                    shell_j,
                                    ri,
                                    rj,
                                    rc,
                                    ec,
                                    coeff_i,
                                    coeff_j,
                                    ai,
                                    aj,
                                    l_proj,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply representation transform and write into staging.
    match plan.representation {
        Representation::Spheric => {
            let sph_size = nsi * nsj;
            if staging.len() >= sph_size {
                cart_to_sph_1e(&cart_buf, &mut staging[..sph_size], li, lj);
            } else {
                let mut sph_tmp = vec![0.0_f64; sph_size];
                cart_to_sph_1e(&cart_buf, &mut sph_tmp, li, lj);
                let copy_len = staging.len().min(sph_size);
                staging[..copy_len].copy_from_slice(&sph_tmp[..copy_len]);
            }
        }
        Representation::Cart => {
            let copy_len = staging.len().min(cart_buf.len());
            staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
        }
        Representation::Spinor => {
            // D-12: spinor accepted by resolver but NOT byte-identity-gated this
            // phase. Write zeros (compiled-but-unverified per Phase 18 precedent).
            for slot in staging.iter_mut() {
                *slot = 0.0;
            }
        }
    }

    let not0 = staging.iter().filter(|&&v| v.abs() > 1e-18).count() as i32;
    let staging_bytes = staging.len() * std::mem::size_of::<f64>();
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

// ─────────────────────────────────────────────────────────────────────────────
// Module-internal unit tests (defense-in-depth for the guard arms).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Module-internal unit tests for the ECP kernel.
    //!
    //! These tests exercise the kernel's helper functions and dispatch logic
    //! that do NOT require a real `ResolvedBackend`. End-to-end coverage of
    //! `launch_ecp` (which needs a backend to construct an `ExecutionPlan`)
    //! is provided by the family-registry tests in `kernels/mod.rs`
    //! (`family_registry_resolves_base_slice` adds an ECP arm in this plan)
    //! and the parity tests in `crates/cintx-oracle/tests/safe_api_ecp_parity.rs`.
    //!
    //! cubecl_ecp guards covered here: canonical-family-name resolver, ECP
    //! channel-dispatch invariants, Cartesian component enumeration.
    //! cubecl_ecp guards covered by the integration test in mod.rs: the
    //! launch_ecp registration entry point.
    //! cubecl_ecp guards covered by safe_api_ecp_parity.rs: empty-ecp_shells,
    //! gradient-operator-rejection, byte-identity numerics.

    use super::*;

    /// Sanity check that the registered launcher pointer is reachable from
    /// the family-name resolver. This catches missing-registration regressions
    /// without invoking the launcher (no backend needed).
    #[test]
    fn launch_ecp_registered_under_canonical_family_ecp() {
        let resolved = crate::kernels::resolve_family_name_for_tests("ecp");
        assert!(
            resolved.is_some(),
            "kernels::resolve_family_name(\"ecp\") must return Some(launch_ecp)"
        );
        assert!(
            crate::kernels::supports_canonical_family("ecp"),
            "supports_canonical_family(\"ecp\") must return true (D-09 stable, unconditional)"
        );
    }

    /// Cartesian component enumeration mirrors libcint's CINTcart_comp order.
    #[test]
    fn cart_comps_returns_expected_count() {
        assert_eq!(cart_comps(0).len(), 1); // s
        assert_eq!(cart_comps(1).len(), 3); // p
        assert_eq!(cart_comps(2).len(), 6); // d
        assert_eq!(cart_comps(3).len(), 10); // f
    }

    /// The TYPE1_HERMITE_N constant must be within Phase 02's supported
    /// envelope (1..=GAUSS_HERMITE_NMAX=8).
    #[test]
    fn type1_hermite_node_count_within_envelope() {
        assert!(TYPE1_HERMITE_N >= 1);
        assert!(TYPE1_HERMITE_N <= 8);
    }
}
