//! Byte-identity parity tests for the per-nucleus ECP force operator
//! `ECPscalar_iprinv` — the two gradient representations
//! (`int1e_ecp_iprinv_cart`, `int1e_ecp_iprinv_sph`, component_rank=3).
//!
//! Targets byte-identity vs vendored PySCF nr_ecp_deriv at the Phase 15 unified
//! tolerance (atol=1e-12, rtol=0.0) on the Cu/LANL2DZ corpus, sweeping the rinv
//! origin over each ECP-bearing atom (21-07, GRAD-09, D-09).
//!
//! ## What `iprinv` is (and how it differs from `ipnuc`)
//!
//! `ECPscalar_iprinv` is the per-nucleus ECP force term: the 3-component
//! `∂/∂A_i^{x,y,z}` derivative of the ECP integral, but restricted to the SINGLE
//! ECP shell on the atom selected by the rinv origin — dropping the all-slot
//! accumulation `ipnuc` performs over every ECP-bearing nucleus.
//!   - vendor side: selects the atom via `env[AS_RINV_ORIG_ATOM]` (an INTEGER
//!     atom index) + `_one_shell_ecpbas` (nr_ecp_deriv.c:333 cart / :420 sph).
//!   - cintx side: selects the atom whose `coord_bohr` matches the rinv-origin
//!     COORDINATE (`ExecutionOptions::rinv_orig`, plumbed in 21-01), within a
//!     tight tolerance (`launch_ecp::IPRINV_ORIGIN_MATCH_TOL = 1e-10`).
//!
//! ## Cu/LANL2DZ is a single-ECP-atom fixture
//!
//! The shipped Cu/LANL2DZ fixture carries the ECP on the Cu atom ONLY (atom
//! index 0). With a single ECP-bearing nucleus the per-nucleus selection
//! degenerates: `iprinv` at the Cu origin sees exactly the same slabs `ipnuc`
//! sums (there is only one). So for THIS fixture `iprinv@Cu == ipnuc` by
//! construction — we assert that tie as an extra cross-check (the selection
//! restricts to the SAME single nucleus ipnuc already covers, never a wrong or
//! extra atom). The KERNEL-level proof that `iprinv` actually restricts (i.e.
//! `iprinv != all-slot ipnuc` when >1 ECP atom exists) lives in the
//! `cintx-cubecl` unit test `iprinv_selects_only_the_origin_atom`
//! (`select_iprinv_slots` returns ONLY the matched nucleus's slot). The
//! byte-identity vs the dedicated `vendor_ECPscalar_iprinv_*` symbol below
//! confirms the per-nucleus path is correct against the C reference directly.
//!
//! Run via:
//!     CINTX_ORACLE_BUILD_VENDOR=1 cargo test --locked \
//!         -p cintx-oracle --features cpu --test ecp_iprinv_parity

// Module gate matches safe_api_ecp_parity.rs.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, AS_ECPBAS_OFFSET, AS_NECPBAS, ATM_SLOTS, BAS_SLOTS, PTR_COORD,
};
use cintx_core::ecp::{EcpChannel, EcpShell};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_oracle::fixtures::build_cu_lanl2dz;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// Phase 15 unified tolerance — every parity test in cintx-oracle uses these.
const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;

// PySCF nr_ecp env slot for the iprinv target atom index (nr_ecp.h:30,
// AS_RINV_ORIG_ATOM = 17). NOT re-exported from cintx_compat::raw (the cintx
// safe API selects by COORDINATE, not index), so we mirror the vendor constant
// locally for the vendor-call env setup.
const AS_RINV_ORIG_ATOM: usize = 17;

// ─────────────────────────────────────────────────────────────────────────────
// Typed Cu/LANL2DZ BasisSet builder (mirrors safe_api_ecp_parity.rs:95-189).
// ─────────────────────────────────────────────────────────────────────────────

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

/// Returns the typed BasisSet, the AO Shell vector, and the molecule's atom
/// coordinates (for the per-nucleus rinv-origin sweep).
fn build_cu_lanl2dz_safe_basis(
    rep: Representation,
) -> (BasisSet, Vec<Arc<Shell>>, Vec<[f64; 3]>) {
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
    let atom_coords = vec![coord];
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
    (basis, shells, atom_coords)
}

/// Resolve the (positionally-paired) `OperatorId` for an ECP iprinv symbol from
/// the manifest descriptors (never hardcode the integer — Phase 19-03 D).
fn iprinv_operator_id(symbol: &str) -> OperatorId {
    Resolver::descriptor_by_symbol(symbol)
        .unwrap_or_else(|e| panic!("missing manifest descriptor for {symbol}: {e:?}"))
        .id
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx safe-API iprinv collector — drives SessionRequest::evaluate over the
// full shell-pair Cartesian product with the rinv origin set to `origin`.
// Output is 3 × n_ao × n_ao, axis-outer F-order [axis, ao_j, ao_i].
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_iprinv_matrix(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
    origin: [f64; 3],
) -> Vec<f64> {
    let shell_nao: Vec<usize> = shells.iter().map(|s| s.ao_per_shell()).collect();
    let n_ao: usize = shell_nao.iter().sum();
    let mut comp_matrix = vec![0.0_f64; 3 * n_ao * n_ao];

    let mut options = ExecutionOptions::default();
    options.rinv_orig = Some(origin);

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];
            let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple");
            let request =
                SessionRequest::new(op, rep, basis, shell_tuple, options.clone());
            let query = request.query_workspace().expect("query_workspace");
            let output = query.evaluate().expect("evaluate");
            let pair = &output.tensor.owned_values;
            assert_eq!(
                pair.len(),
                3 * ni * nj,
                "iprinv safe-API pair buffer must be 3*ni*nj (component_rank=3)"
            );
            for axis in 0..3 {
                for ii in 0..ni {
                    for jj in 0..nj {
                        // F-order [axis, ao_j, ao_i]: axis slowest, ao_i fastest.
                        let v = pair[axis * ni * nj + jj * ni + ii];
                        comp_matrix[axis * n_ao * n_ao
                            + (row_offset + ii) * n_ao
                            + (col_offset + jj)] = v;
                    }
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    comp_matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor PySCF nr_ecp_deriv iprinv collector — packs (atom_bas ++ ecp_bas) and
// sets env[AS_ECPBAS_OFFSET]/AS_NECPBAS (slab packing) AND
// env[AS_RINV_ORIG_ATOM]=atom_id (the per-nucleus selector). Mirrors
// collect_ecp_ipnuc_matrix_vendor (safe_api_ecp_parity.rs:460-546).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn collect_iprinv_matrix_vendor(
    rep: &str,
    atm: &[i32],
    bas: &[i32],
    ecpbas: &[i32],
    env: &[f64],
    atom_id: i32,
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let nbas_ao = (bas.len() / BAS_SLOTS) as i32;
    let necpbas = (ecpbas.len() / BAS_SLOTS) as i32;

    let mut combined_bas = Vec::with_capacity(bas.len() + ecpbas.len());
    combined_bas.extend_from_slice(bas);
    combined_bas.extend_from_slice(ecpbas);
    let combined_nbas = nbas_ao + necpbas;

    let mut env_with_ecp = env.to_vec();
    while env_with_ecp.len() <= AS_NECPBAS.max(AS_RINV_ORIG_ATOM) {
        env_with_ecp.push(0.0);
    }
    env_with_ecp[AS_ECPBAS_OFFSET] = nbas_ao as f64;
    env_with_ecp[AS_NECPBAS] = necpbas as f64;
    // The per-nucleus selector: pick the ECP shell on `atom_id`.
    env_with_ecp[AS_RINV_ORIG_ATOM] = atom_id as f64;

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

    let mut comp_matrix = vec![0.0_f64; 3 * n_ao * n_ao];
    let mut row_offset = 0usize;
    for si in 0..nbas_ao as usize {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..nbas_ao as usize {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; 3 * ni * nj];

            let _ret = match rep {
                "sph" => vendor_ffi::vendor_ECPscalar_iprinv_sph(
                    &mut out,
                    &shls,
                    atm,
                    natm,
                    &combined_bas,
                    combined_nbas,
                    &env_with_ecp,
                ),
                "cart" => vendor_ffi::vendor_ECPscalar_iprinv_cart(
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

            // PySCF [comp, dij] with dij F-order j*ni+i (i fastest). No transpose.
            for axis in 0..3 {
                for ii in 0..ni {
                    for jj in 0..nj {
                        let v = out[axis * ni * nj + jj * ni + ii];
                        comp_matrix[axis * n_ao * n_ao
                            + (row_offset + ii) * n_ao
                            + (col_offset + jj)] = v;
                    }
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    comp_matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance helper — copied verbatim from safe_api_ecp_parity.rs:326-341.
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

/// Read the per-atom coordinate from the raw `atm`/`env` slab (PTR_COORD slot
/// in each ATM_SLOTS-wide row points into `env`).
fn atom_coord_from_slab(atm: &[i32], env: &[f64], atom_id: usize) -> [f64; 3] {
    let base = atm[atom_id * ATM_SLOTS + PTR_COORD] as usize;
    [env[base], env[base + 1], env[base + 2]]
}

/// Indices of ECP-bearing atoms in the Cu/LANL2DZ slab. The ECP rows live in the
/// separate `ecpbas` slab; their ATOM_OF column is the atom index. We sweep the
/// distinct atom indices that host an ECP shell.
fn ecp_bearing_atoms(ecpbas: &[i32]) -> Vec<usize> {
    use cintx_compat::raw::ATOM_OF;
    let mut atoms: Vec<usize> = ecpbas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ATOM_OF] as usize)
        .collect();
    atoms.sort_unstable();
    atoms.dedup();
    atoms
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-symbol byte-identity tests — sweep the rinv origin over each ECP-bearing
// atom, asserting 0 mismatches at atol=1e-12 vs vendored PySCF nr_ecp_deriv.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
#[allow(non_snake_case)] // mirrors the vendor PySCF symbol name for traceability
fn test_ECPscalar_iprinv_cart_cu_lanl2dz_parity() {
    let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
    let (basis, shells, _coords) = build_cu_lanl2dz_safe_basis(Representation::Cart);
    let op = iprinv_operator_id("int1e_ecp_iprinv_cart");

    let ecp_atoms = ecp_bearing_atoms(&ecpbas);
    assert!(!ecp_atoms.is_empty(), "Cu/LANL2DZ must have ≥1 ECP-bearing atom");

    let mut any_nonzero = false;
    for &atom_id in &ecp_atoms {
        let origin = atom_coord_from_slab(&atm, &env, atom_id);

        let safe_matrix =
            collect_safe_api_iprinv_matrix(op, Representation::Cart, &basis, &shells, origin);
        let vendor_matrix =
            collect_iprinv_matrix_vendor("cart", &atm, &bas, &ecpbas, &env, atom_id as i32);

        let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, ATOL, RTOL);
        assert_eq!(
            mismatches, 0,
            "ECPscalar_iprinv_cart @ atom {atom_id}: {mismatches} elements exceed \
             atol={ATOL:.0e}/rtol={RTOL:.0e} vs vendored PySCF nr_ecp_deriv (3 components)"
        );
        if safe_matrix.iter().any(|v| v.abs() > 1e-12) {
            any_nonzero = true;
        }
    }
    assert!(
        any_nonzero,
        "iprinv cart output was all-zero across every ECP atom — the parity gate would be vacuous"
    );
}

#[test]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
#[allow(non_snake_case)] // mirrors the vendor PySCF symbol name for traceability
fn test_ECPscalar_iprinv_sph_cu_lanl2dz_parity() {
    let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
    let (basis, shells, _coords) = build_cu_lanl2dz_safe_basis(Representation::Spheric);
    let op = iprinv_operator_id("int1e_ecp_iprinv_sph");

    let ecp_atoms = ecp_bearing_atoms(&ecpbas);
    assert!(!ecp_atoms.is_empty(), "Cu/LANL2DZ must have ≥1 ECP-bearing atom");

    let mut any_nonzero = false;
    for &atom_id in &ecp_atoms {
        let origin = atom_coord_from_slab(&atm, &env, atom_id);

        let safe_matrix =
            collect_safe_api_iprinv_matrix(op, Representation::Spheric, &basis, &shells, origin);
        let vendor_matrix =
            collect_iprinv_matrix_vendor("sph", &atm, &bas, &ecpbas, &env, atom_id as i32);

        let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, ATOL, RTOL);
        assert_eq!(
            mismatches, 0,
            "ECPscalar_iprinv_sph @ atom {atom_id}: {mismatches} elements exceed \
             atol={ATOL:.0e}/rtol={RTOL:.0e} vs vendored PySCF nr_ecp_deriv (3 components)"
        );
        if safe_matrix.iter().any(|v| v.abs() > 1e-12) {
            any_nonzero = true;
        }
    }
    assert!(
        any_nonzero,
        "iprinv sph output was all-zero across every ECP atom — the parity gate would be vacuous"
    );
}

/// Single-ECP-atom tie cross-check (see file rustdoc): on Cu/LANL2DZ the ECP is
/// on Cu ONLY, so `iprinv@Cu` must equal `ipnuc` (the all-slot sum over a single
/// slot). This proves the per-nucleus selection restricts to the SAME single
/// nucleus ipnuc covers — never the wrong atom, never an extra one. The
/// >1-ECP-atom restriction is unit-tested in cintx-cubecl
/// (`iprinv_selects_only_the_origin_atom`).
#[test]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn iprinv_at_cu_equals_ipnuc_for_single_ecp_atom_fixture() {
    let (atm, _bas, ecpbas, env) = build_cu_lanl2dz();
    let (basis, shells, _coords) = build_cu_lanl2dz_safe_basis(Representation::Cart);

    let ecp_atoms = ecp_bearing_atoms(&ecpbas);
    assert_eq!(
        ecp_atoms.len(),
        1,
        "this tie cross-check assumes the single-ECP-atom Cu/LANL2DZ fixture"
    );
    let cu = ecp_atoms[0];
    let origin = atom_coord_from_slab(&atm, &env, cu);

    let iprinv = collect_safe_api_iprinv_matrix(
        iprinv_operator_id("int1e_ecp_iprinv_cart"),
        Representation::Cart,
        &basis,
        &shells,
        origin,
    );
    let ipnuc = collect_safe_api_iprinv_matrix(
        OperatorId::INT1E_ECP_IPNUC_CART,
        Representation::Cart,
        &basis,
        &shells,
        origin, // ipnuc ignores the origin (sums all slots); harmless to pass it
    );

    let mismatches = count_mismatches(&ipnuc, &iprinv, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "single-ECP-atom fixture: iprinv@Cu must equal ipnuc ({mismatches} mismatches)"
    );
}
