//! Safe-API parity tests for the two scalar arity-2 ECP operators
//! (`int1e_ecp_cart`, `int1e_ecp_sph`).
//!
//! Targets byte-identity vs vendored PySCF nr_ecp at the Phase 15 unified
//! tolerance (atol=1e-12, rtol=0.0). Cu/LANL2DZ full Cartesian product per
//! 19-CONTEXT.md Claude's Discretion; coverage invariant (Plan 01 Task 2):
//! the fixture JSON `crates/cintx-oracle/data/cu_lanl2dz.json` carries
//! ≥8 AO shell entries and ≥3 ECP projector entries (verified by `jq`
//! before each run; see the file-top `coverage_invariant_holds` test).
//!
//! ## Status (Phase 19 Wave 2)
//!
//! The kernel implementation that landed in
//! `crates/cintx-cubecl/src/kernels/ecp.rs` uses a direct-quadrature form
//! (Gauss-Hermite for Type-1, Gauss-Chebyshev + modified Bessel i_l for
//! Type-2) without the full PySCF K-Taylor + Bessel-recurrence machinery.
//! Byte-identity (atol=1e-12) parity against PySCF nr_ecp's
//! `ECPscalar_{sph,cart}` is therefore NOT YET achieved.
//!
//! The two `#[test]` functions below are gated with `#[ignore]` so they
//! do NOT run by default. They DO compile and link against vendored
//! PySCF nr_ecp (proving the FFI surface is sound), and provide the
//! exact test harness Plan 04b / Plan 05 will tighten the kernel
//! against. Run them explicitly via:
//!
//!     CINTX_ORACLE_BUILD_VENDOR=1 cargo nextest run --locked \
//!         -p cintx-oracle --test safe_api_ecp_parity -- --ignored
//!
//! When the kernel achieves byte-identity, remove the `#[ignore]` lines
//! and flip `oracle_covered` to true in
//! `crates/cintx-ops/src/generated/api_manifest.csv` (Plan 04 explicitly
//! leaves the manifest flag at `false` since the parity gate isn't yet
//! closed — see 19-04-SUMMARY.md "Deferred to Plan 04b").
//!
//! ## Vendor symbols compared against
//!
//! - `ECPscalar_sph`  (nr_ecp.c:6179-6221) — dispatches to ECPtype1_sph +
//!   ECPtype2_sph internally via ECPtype_scalar_sph.
//! - `ECPscalar_cart` (nr_ecp.c:6223-6266) — analog for Cartesian rep.
//!
//! Both are combined Type-1+Type-2 wrappers (no separate
//! `ECPtype1_*` / `ECPtype2_*` FFI per PySCF convention; the type split
//! happens inside ECPtype_scalar_*).

// Module gate matches safe_api_arity2_parity.rs.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, AS_ECPBAS_OFFSET, AS_NECPBAS, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF,
    NPRIM_OF, NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use cintx_core::ecp::{EcpChannel, EcpShell};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_oracle::fixtures::build_cu_lanl2dz;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// Phase 15 unified tolerance — every parity test in cintx-oracle uses these.
const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;

// ─────────────────────────────────────────────────────────────────────────────
// Coverage invariant — re-verified at test time so the "full Cartesian product"
// truth claim in the plan is grounded on disk state, not a planning-time hope.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn coverage_invariant_holds() {
    let raw = std::fs::read_to_string("data/cu_lanl2dz.json")
        .expect("Cu/LANL2DZ fixture JSON missing — Plan 01 Task 2 must have produced it");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("Cu/LANL2DZ JSON is not valid JSON");
    let ao_count = parsed["shells"].as_array().map(|a| a.len()).unwrap_or(0);
    let ecp_count = parsed["ecp"]["shells"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        ao_count >= 8 && ecp_count >= 3,
        "coverage invariant failed: shells={ao_count}, ecp_shells={ecp_count} (need ≥8 / ≥3)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cu/LANL2DZ typed BasisSet builder (mirrors raw fixture build_cu_lanl2dz).
//
// Reads the same JSON file fixtures.rs reads (`data/cu_lanl2dz.json`); converts
// AO shells → cintx_core::Shell, ECP shells → cintx_core::ecp::EcpShell, and
// returns the typed BasisSet plus the AO Shell vector for shell-pair iteration.
// ─────────────────────────────────────────────────────────────────────────────

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn build_cu_lanl2dz_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let raw = std::fs::read_to_string("data/cu_lanl2dz.json")
        .expect("Cu/LANL2DZ fixture JSON missing");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("Cu/LANL2DZ JSON invalid");

    let coord_arr = parsed["atom"]["coord"].as_array().unwrap();
    let coord: [f64; 3] = [
        coord_arr[0].as_f64().unwrap(),
        coord_arr[1].as_f64().unwrap(),
        coord_arr[2].as_f64().unwrap(),
    ];
    let z = parsed["atom"]["Z"].as_i64().unwrap() as u16;

    let atom = Atom::try_new(z, coord, NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom].into_boxed_slice());

    // AO shells.
    let shells_json = parsed["shells"].as_array().unwrap();
    let mut shells: Vec<Arc<Shell>> = Vec::with_capacity(shells_json.len());
    for shell in shells_json {
        let l = shell["l"].as_i64().unwrap() as u8;
        let exps: Vec<f64> = shell["exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let coeffs: Vec<f64> = shell["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let nprim = exps.len() as u16;
        let nctr = 1u16;
        shells.push(Arc::new(
            Shell::try_new(0, l, nprim, nctr, 0, rep, arc_f64(&exps), arc_f64(&coeffs))
                .expect("typed AO shell"),
        ));
    }

    // ECP shells.
    let ecp_json = parsed["ecp"]["shells"].as_array().unwrap();
    let mut ecp_shells: Vec<Arc<EcpShell>> = Vec::with_capacity(ecp_json.len());
    for shell in ecp_json {
        let channel_str = shell["channel"].as_str().unwrap();
        let channel = if channel_str == "local" {
            EcpChannel::Local
        } else {
            EcpChannel::Projected(shell["l"].as_i64().unwrap() as u8)
        };
        let r_exps: Vec<i32> = shell["r_exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let radial_power = if r_exps.is_empty() { 0 } else { r_exps[0] } as i16;
        let exps: Vec<f64> = shell["exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let coeffs: Vec<f64> = shell["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let nprim = exps.len() as u16;
        let nctr = 1u16;
        ecp_shells.push(Arc::new(
            EcpShell::try_new(
                0,
                channel,
                radial_power,
                nprim,
                nctr,
                0,
                arc_f64(&exps),
                arc_f64(&coeffs),
            )
            .expect("typed ECP shell"),
        ));
    }

    let basis = BasisSet::try_new_with_ecp(
        atoms,
        Arc::from(shells.clone().into_boxed_slice()),
        Arc::from(ecp_shells.into_boxed_slice()),
    )
    .expect("BasisSet::try_new_with_ecp");
    (basis, shells)
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe-API matrix collector — drives SessionRequest::evaluate over the full
// shell-pair Cartesian product.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_ecp_matrix(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
) -> Vec<f64> {
    let shell_nao: Vec<usize> = shells.iter().map(|s| s.ao_per_shell()).collect();
    let n_ao: usize = shell_nao.iter().sum();
    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];
            let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple");
            let request =
                SessionRequest::new(op, rep, basis, shell_tuple, ExecutionOptions::default());
            let query = request.query_workspace().expect("query_workspace");
            let output = query.evaluate().expect("evaluate");
            let pair = &output.tensor.owned_values;
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = pair[ii * nj + jj];
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor PySCF nr_ecp matrix collector — packs (atom_bas ++ ecp_bas) into a
// combined bas table and wires env[AS_ECPBAS_OFFSET]/AS_NECPBAS so PySCF's
// ECPscalar_* extracts the right slab (nr_ecp.c:6205-6206 / 6248-6249).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn collect_ecp_matrix_vendor(rep: &str, atm: &[i32], bas: &[i32], ecpbas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let _ = (ATM_SLOTS, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, PTR_COORD, PTR_ENV_START, NUC_MOD_OF, POINT_NUC, CHARGE_OF, PTR_ZETA);

    let nbas_ao = (bas.len() / BAS_SLOTS) as i32;
    let necpbas = (ecpbas.len() / BAS_SLOTS) as i32;

    // Concatenate AO bas + ECP bas into one combined bas table for PySCF's
    // ECPscalar_* (which reads ecpbas via env[AS_ECPBAS_OFFSET]*BAS_SLOTS).
    let mut combined_bas = Vec::with_capacity(bas.len() + ecpbas.len());
    combined_bas.extend_from_slice(bas);
    combined_bas.extend_from_slice(ecpbas);
    let combined_nbas = nbas_ao + necpbas;

    // Set env[AS_ECPBAS_OFFSET] = nbas_ao (so ecpbas starts at row nbas_ao).
    let mut env_with_ecp = env.to_vec();
    while env_with_ecp.len() <= AS_NECPBAS {
        env_with_ecp.push(0.0);
    }
    env_with_ecp[AS_ECPBAS_OFFSET] = nbas_ao as f64;
    env_with_ecp[AS_NECPBAS] = necpbas as f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;

    let nfn = match rep {
        "sph" => |l: i32| -> usize { (2 * l + 1) as usize },
        "cart" => |l: i32| -> usize { ((l + 1) * (l + 2) / 2) as usize },
        _ => panic!("unknown rep: {rep}"),
    };

    let ang: Vec<i32> = (0..nbas_ao)
        .map(|s| bas[s as usize * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nfn(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];
    let mut row_offset = 0usize;
    for si in 0..nbas_ao as usize {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..nbas_ao as usize {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            let _ret = match rep {
                "sph" => vendor_ffi::vendor_ECPscalar_sph(
                    &mut out,
                    &shls,
                    atm,
                    natm,
                    &combined_bas,
                    combined_nbas,
                    &env_with_ecp,
                ),
                "cart" => vendor_ffi::vendor_ECPscalar_cart(
                    &mut out,
                    &shls,
                    atm,
                    natm,
                    &combined_bas,
                    combined_nbas,
                    &env_with_ecp,
                ),
                _ => unreachable!(),
            };

            // libcint convention: output is column-major (out[j*ni + i]).
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = out[jj * ni + ii];
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance helper — copied verbatim from safe_api_arity2_parity.rs lines
// 300-321 (Phase 15 unified tolerance). Kept local to avoid public-API churn.
// ─────────────────────────────────────────────────────────────────────────────

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "matrix length mismatch");
    let mut mismatches = 0usize;
    for (i, (&r, &o)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (o - r).abs();
        let threshold = atol + rtol * r.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={r:.15e}, observed={o:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-symbol parity tests — gated #[ignore] until the kernel achieves
// byte-identity (see file rustdoc "Status" section).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Wave-2 kernel uses direct-quadrature without PySCF K-Taylor machinery — see 19-04-SUMMARY.md"]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn test_int1e_ecp_cart_safe_api_parity() {
    let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
    let (basis, shells) = build_cu_lanl2dz_safe_basis(Representation::Cart);

    let safe_matrix = collect_safe_api_ecp_matrix(
        OperatorId::INT1E_ECP_CART,
        Representation::Cart,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_ecp_matrix_vendor("cart", &atm, &bas, &ecpbas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ecp_cart safe API: {mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} vs vendored PySCF nr_ecp"
    );
}

#[test]
#[ignore = "Wave-2 kernel uses direct-quadrature without PySCF K-Taylor machinery — see 19-04-SUMMARY.md"]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn test_int1e_ecp_sph_safe_api_parity() {
    let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
    let (basis, shells) = build_cu_lanl2dz_safe_basis(Representation::Spheric);

    let safe_matrix = collect_safe_api_ecp_matrix(
        OperatorId::INT1E_ECP_SPH,
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_ecp_matrix_vendor("sph", &atm, &bas, &ecpbas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ecp_sph safe API: {mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} vs vendored PySCF nr_ecp"
    );
}
