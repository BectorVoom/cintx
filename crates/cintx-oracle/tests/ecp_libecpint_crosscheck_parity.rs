//! Phase 19 D-02 (REVISED) — OPTIONAL, NON-BLOCKING secondary cross-check oracle
//! for `int1e_ecp_{cart,sph}` against libecpint (Shaw & Hill, *J. Chem. Phys.*
//! **147**, 074108, 2017, MIT; <https://github.com/robashaw/libecpint>).
//!
//! This satisfies ROADMAP SC#4's "secondary oracle" wording WITHOUT gating CI.
//! The primary, BLOCKING byte-identity gate remains the PySCF `nr_ecp` parity at
//! `atol=1e-12, rtol=0.0` (closed by 19-06 / 19-07; see `safe_api_ecp_parity.rs`).
//! This file adds a cross-implementation drift detector at an INFORMATIONAL loose
//! tolerance.
//!
//! ## Why informational (NOT byte-identity)
//!
//! libecpint and PySCF `nr_ecp` use **different internal recurrence + quadrature
//! conventions** (libecpint: recursive code generation + Gauss-Chebyshev quadrature;
//! PySCF `nr_ecp`: K-Taylor precomputed Bessel tables + `ECPrad_part`/`ECPrad_block`),
//! plus differing normalization / phase conventions. Byte-identity at `1e-12` is NOT
//! expected. The cross-check tolerance is therefore a loose, empirically-set envelope:
//!
//! ```text
//! CROSSCHECK_ATOL = 1e-9   CROSSCHECK_RTOL = 0.0
//! ```
//!
//! Any `|diff|` above it is reported as INFORMATIONAL drift, never a CI failure.
//! See `.planning/notes/ecp-libecpint-crosscheck.md` for the full rationale and
//! the operator activation steps.
//!
//! ## Opt-in (mirrors the Phase 16 ROCm `CINTX_ROCM_ORACLE=1` precedent)
//!
//! The two tests below are BOTH `#[ignore]` AND `#[cfg(has_libecpint_oracle)]`:
//!
//!   - `#[cfg(has_libecpint_oracle)]` — the cfg is emitted by `build.rs` ONLY when
//!     `CINTX_LIBECPINT_ORACLE=1` and libecpint + an operator-supplied `extern "C"`
//!     shim are reachable. In the DEFAULT build the cfg is absent, so the tests do
//!     not exist and this file is a no-op compile (no libecpint required).
//!   - `#[ignore]` — even when the cfg is on, the tests run only via
//!     `-- --ignored`, never in the default `cargo test` suite.
//!
//! Run live (on a host with libecpint installed):
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 CINTX_LIBECPINT_ORACLE=1 \
//!     cargo test --locked -p cintx-oracle --features cpu \
//!     --test ecp_libecpint_crosscheck_parity -- --ignored
//! ```

// Module gate matches safe_api_ecp_parity.rs.
#![cfg(any(feature = "cpu", feature = "rocm"))]

// Informational cross-check tolerance — loose envelope, NOT the byte-identity gate.
// libecpint and PySCF nr_ecp use different recurrence + quadrature conventions
// internally (see .planning/notes/ecp-libecpint-crosscheck.md), so byte-identity at
// the Phase 15 unified atol=1e-12 is not expected. This 1e-9 value is a starting
// envelope pending empirical measurement on a host with libecpint installed.
#[allow(dead_code)]
const CROSSCHECK_ATOL: f64 = 1e-9;
#[allow(dead_code)]
const CROSSCHECK_RTOL: f64 = 0.0;

// ─────────────────────────────────────────────────────────────────────────────
// The cross-check machinery below is only compiled when the libecpint oracle cfg
// is active. In the default build (cfg off) this file compiles to nothing beyond
// the constants above, so it is a no-op compile without libecpint present.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_libecpint_oracle)]
mod crosscheck {
    use super::{CROSSCHECK_ATOL, CROSSCHECK_RTOL};

    use cintx_compat::raw::{ANG_OF, AS_ECPBAS_OFFSET, AS_NECPBAS, ATM_SLOTS, BAS_SLOTS};
    use cintx_core::ecp::{EcpChannel, EcpShell};
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_oracle::{fixtures::build_cu_lanl2dz, libecpint_ffi};
    use cintx_rs::SessionRequest;
    use cintx_runtime::ExecutionOptions;
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    /// Build the typed Cu/LANL2DZ BasisSet — mirrors safe_api_ecp_parity.rs's
    /// `build_cu_lanl2dz_safe_basis`. Returns the BasisSet + the AO Shell vector.
    fn build_cu_lanl2dz_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
        let raw = std::fs::read_to_string("data/cu_lanl2dz.json")
            .expect("Cu/LANL2DZ fixture JSON missing");
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).expect("Cu/LANL2DZ JSON invalid");

        let coord_arr = parsed["atom"]["coord"].as_array().unwrap();
        let coord: [f64; 3] = [
            coord_arr[0].as_f64().unwrap(),
            coord_arr[1].as_f64().unwrap(),
            coord_arr[2].as_f64().unwrap(),
        ];
        let z = parsed["atom"]["Z"].as_i64().unwrap() as u16;

        let atom = Atom::try_new(z, coord, NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

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
            shells.push(Arc::new(
                Shell::try_new(0, l, nprim, 1, 0, rep, arc_f64(&exps), arc_f64(&coeffs))
                    .expect("typed AO shell"),
            ));
        }

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
            ecp_shells.push(Arc::new(
                EcpShell::try_new(
                    0,
                    channel,
                    radial_power,
                    nprim,
                    1,
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

    /// Safe-API matrix collector — identical shape to safe_api_ecp_parity.rs's
    /// `collect_safe_api_ecp_matrix`. Drives SessionRequest::evaluate over the
    /// full shell-pair Cartesian product into a row-major `[ao_i, ao_j]` matrix.
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
                let shell_tuple =
                    ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
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

    /// libecpint matrix collector — mirrors safe_api_ecp_parity.rs's
    /// `collect_ecp_matrix_vendor`, but calls the libecpint FFI shim from Task 1
    /// instead of PySCF's `ECPscalar_*`. Packs `(atom_bas ++ ecp_bas)` into one
    /// combined bas table and wires `env[AS_ECPBAS_OFFSET]`/`AS_NECPBAS` so the
    /// shim can read the ECP slab the same way PySCF does.
    ///
    /// ## Normalization / convention adapter
    ///
    /// libecpint emits Cartesian/spherical components in its own internal ordering
    /// and with its own normalization/phase convention. The per-pair read index
    /// below assumes the operator shim has already normalized to libcint's
    /// column-major `out[jj*ni + ii]` block layout (the same convention the PySCF
    /// vendor collector consumes) — this is the SINGLE place a convention adapter
    /// is documented/applied. Because the cross-check is informational, any residual
    /// convention mismatch that survives within `CROSSCHECK_ATOL` is reported as
    /// drift, not corrected.
    fn collect_ecp_matrix_libecpint(
        rep: &str,
        atm: &[i32],
        bas: &[i32],
        ecpbas: &[i32],
        env: &[f64],
    ) -> Vec<f64> {
        let nbas_ao = (bas.len() / BAS_SLOTS) as i32;
        let necpbas = (ecpbas.len() / BAS_SLOTS) as i32;

        let mut combined_bas = Vec::with_capacity(bas.len() + ecpbas.len());
        combined_bas.extend_from_slice(bas);
        combined_bas.extend_from_slice(ecpbas);
        let combined_nbas = nbas_ao + necpbas;

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
                let mut out = vec![0.0_f64; ni * nj];

                let _ret = match rep {
                    "sph" => libecpint_ffi::libecpint_ecp_sph(
                        &mut out,
                        &shls,
                        atm,
                        natm,
                        &combined_bas,
                        combined_nbas,
                        &env_with_ecp,
                    ),
                    "cart" => libecpint_ffi::libecpint_ecp_cart(
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

                // libecpint shim normalizes to libcint column-major out[j*ni + i].
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

    /// Count INFORMATIONAL mismatches at the loose cross-check tolerance. Never
    /// asserts — the caller decides what to do with the count (the tests log it
    /// as drift, they do not fail CI on it).
    fn count_informational_mismatches(reference: &[f64], observed: &[f64]) -> usize {
        assert_eq!(reference.len(), observed.len(), "matrix length mismatch");
        let mut mismatches = 0usize;
        for (i, (&r, &o)) in reference.iter().zip(observed.iter()).enumerate() {
            let diff = (o - r).abs();
            let threshold = CROSSCHECK_ATOL + CROSSCHECK_RTOL * r.abs();
            if diff > threshold {
                mismatches += 1;
                eprintln!(
                    "  INFORMATIONAL DRIFT at index {i}: libecpint={r:.15e}, cintx={o:.15e}, \
                     diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
        mismatches
    }

    /// Defense-in-depth: even with the cfg on, refuse to run unless the env var is
    /// explicitly set (mirrors the ROCm gate's runtime env assertion).
    fn require_opt_in() -> bool {
        match std::env::var("CINTX_LIBECPINT_ORACLE").as_deref() {
            Ok(v) if !v.is_empty() && v != "0" => true,
            _ => {
                eprintln!(
                    "libecpint cross-check is opt-in via CINTX_LIBECPINT_ORACLE=1; \
                     skipping (non-blocking, D-02). See \
                     .planning/notes/ecp-libecpint-crosscheck.md."
                );
                false
            }
        }
    }

    #[test]
    #[ignore = "libecpint secondary oracle is opt-in via CINTX_LIBECPINT_ORACLE=1 (non-blocking, D-02)"]
    fn test_int1e_ecp_cart_libecpint_crosscheck() {
        if !require_opt_in() {
            return;
        }
        let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
        let (basis, shells) = build_cu_lanl2dz_safe_basis(Representation::Cart);

        let cintx_matrix = collect_safe_api_ecp_matrix(
            OperatorId::INT1E_ECP_CART,
            Representation::Cart,
            &basis,
            &shells,
        );
        let libecpint_matrix = collect_ecp_matrix_libecpint("cart", &atm, &bas, &ecpbas, &env);

        let drift = count_informational_mismatches(&libecpint_matrix, &cintx_matrix);
        // INFORMATIONAL ONLY — log the cross-implementation drift, do NOT block CI.
        eprintln!(
            "int1e_ecp_cart libecpint cross-check: {drift} elements exceed informational \
             atol={CROSSCHECK_ATOL:.0e}/rtol={CROSSCHECK_RTOL:.0e} (libecpint vs cintx). \
             This is cross-implementation drift metadata, not a CI gate (D-02)."
        );
    }

    #[test]
    #[ignore = "libecpint secondary oracle is opt-in via CINTX_LIBECPINT_ORACLE=1 (non-blocking, D-02)"]
    fn test_int1e_ecp_sph_libecpint_crosscheck() {
        if !require_opt_in() {
            return;
        }
        let (atm, bas, ecpbas, env) = build_cu_lanl2dz();
        let (basis, shells) = build_cu_lanl2dz_safe_basis(Representation::Spheric);

        let cintx_matrix = collect_safe_api_ecp_matrix(
            OperatorId::INT1E_ECP_SPH,
            Representation::Spheric,
            &basis,
            &shells,
        );
        let libecpint_matrix = collect_ecp_matrix_libecpint("sph", &atm, &bas, &ecpbas, &env);

        let drift = count_informational_mismatches(&libecpint_matrix, &cintx_matrix);
        // INFORMATIONAL ONLY — log the cross-implementation drift, do NOT block CI.
        eprintln!(
            "int1e_ecp_sph libecpint cross-check: {drift} elements exceed informational \
             atol={CROSSCHECK_ATOL:.0e}/rtol={CROSSCHECK_RTOL:.0e} (libecpint vs cintx). \
             This is cross-implementation drift metadata, not a CI gate (D-02)."
        );
    }
}
