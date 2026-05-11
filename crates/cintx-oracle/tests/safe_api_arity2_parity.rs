//! Safe-API parity tests for all 12 arity-2 operators currently routed by `SessionRequest::evaluate`.
//!
//! Eight cart/sph tests assert byte-identity vs vendored libcint 6.1.3 at the Phase 15 unified
//! tolerance (atol=1e-12, rtol=0.0). Four spinor tests use idempotency-only verification
//! because the safe-API spinor output is real-valued (`Vec<f64>`) whereas the vendor spinor
//! FFI wrappers return complex interleaved (re/im) pairs — direct comparison is not meaningful
//! without a complex→real projection, which is deferred to a follow-up phase.
//!
//! See `.planning/phases/17-real-integral-evaluation-in-safe-api/17-CONTEXT.md` for decision
//! rationale (D-06 through D-10).

// Module gate widened to allow `--features rocm` (without cpu) per Phase 16-04 pattern.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G raw fixture (PTR_ENV_START-aware version from center_2c2e_parity.rs)
//
// CRITICAL: libcint reserves env[0..PTR_ENV_START] for global parameters.
// User data MUST start at PTR_ENV_START=20. This version is safe for all 12
// operators including 2c2e, which requires PTR_RANGE_OMEGA (env[8]) to be 0.
// The one_electron_parity.rs version (which starts at env[0]) is NOT used here
// (per 17-PATTERNS.md anti-patterns: it corrupts 2c2e range-separated kernels).
// ─────────────────────────────────────────────────────────────────────────────

const N_SHELLS: usize = 5;

fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];

    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // Layout (starting at PTR_ENV_START=20):
    //   [20..22]  O coords (x, y, z)
    //   [23..25]  H1 coords (x, y, z)
    //   [26..28]  H2 coords (x, y, z)
    //   [29]      PTR_ZETA placeholder (0.0, unused for POINT_NUC)
    //   [30..32]  O 1s exponents
    //   [33..35]  O 1s coefficients
    //   [36..38]  O 2s exponents
    //   [39..41]  O 2s coefficients
    //   [42..44]  O 2p exponents
    //   [45..47]  O 2p coefficients
    //   [48..50]  H 1s exponents
    //   [51..53]  H 1s coefficients
    let mut env = vec![0.0_f64; PTR_ENV_START]; // zeros for reserved slots

    let o_coord_ptr = env.len() as i32;   // 20
    env.extend_from_slice(&o_coord);

    let h1_coord_ptr = env.len() as i32;  // 23
    env.extend_from_slice(&h1_coord);

    let h2_coord_ptr = env.len() as i32;  // 26
    env.extend_from_slice(&h2_coord);

    let zeta_ptr = env.len() as i32;      // 29
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32;   // 30
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; // 33
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32;   // 36
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; // 39
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32;   // 42
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; // 45
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32;   // 48
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; // 51
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];

    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    // Shell 0: O 1s (l=0, 3 prim, 1 ctr)
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    // Shell 1: O 2s (l=0, 3 prim, 1 ctr)
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    // Shell 2: O 2p (l=1, 3 prim, 1 ctr)
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    // Shell 3: H1 1s (l=0, 3 prim, 1 ctr)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    // Shell 4: H2 1s (l=0, 3 prim, 1 ctr)
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G safe-API BasisSet construction helper
//
// Numeric values match build_h2o_sto3g() exactly (same source: Hehre/Stewart/Pople 1969).
// Returns (BasisSet, Vec<Arc<Shell>>) — the caller builds ShellTuple per-call.
// ─────────────────────────────────────────────────────────────────────────────

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o  = Atom::try_new(8, [0.0, 0.0, 0.0],        NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078],  NuclearModel::Point, None, None).unwrap();
    let atom_h2 = Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G exponents and coefficients (Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657, 1969)
    // These values must match build_h2o_sto3g() exactly so vendor comparisons are valid.

    // O 1s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
    let shell_o1s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    // O 2s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
    let shell_o2s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[-0.09996723, 0.39951283, 0.70011547]),
    ).unwrap());

    // O 2p: atom_idx=0, l=1, nprim=3, nctr=1, kappa=0
    let shell_o2p = Arc::new(Shell::try_new(
        0, 1, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[0.15591627, 0.60768372, 0.39195739]),
    ).unwrap());

    // H1 1s: atom_idx=1, l=0, nprim=3, nctr=1, kappa=0
    let shell_h1_1s = Arc::new(Shell::try_new(
        1, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    // H2 1s: atom_idx=2, l=0, nprim=3, nctr=1, kappa=0
    let shell_h2_1s = Arc::new(Shell::try_new(
        2, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    let shells = vec![shell_o1s, shell_o2s, shell_o2p, shell_h1_1s, shell_h2_1s];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe-API matrix collector
//
// Drives `SessionRequest::evaluate` — the safe-API path that Plan 17-02 wired
// to the real CubeClExecutor. This is the system under test.
//
// For an arity-2 operator, the `ShellTuple` must contain exactly 2 shells (one
// bra shell, one ket shell). We iterate over all shell pairs in the basis and
// assemble the full (n_ao × n_ao) matrix — mirroring how `collect_1e_sph_matrix`
// loops over shell pairs when using `eval_raw`. The safe API returns each pair's
// block in row-major order; we insert it into the assembled matrix at the correct
// (row_offset, col_offset) position.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_matrix(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
) -> Vec<f64> {
    // Compute the total number of AOs from shell angular momenta.
    let shell_nao: Vec<usize> = shells
        .iter()
        .map(|s| s.ao_per_shell())
        .collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];

            // Build a 2-shell ShellTuple for this (bra, ket) pair.
            // The safe API is arity-2: ShellTuple must have exactly 2 shells.
            let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple construction must succeed for a valid 2-shell pair");

            let request = SessionRequest::new(
                operator_id,
                rep,
                basis,
                shell_tuple,
                ExecutionOptions::default(),
            );
            let query = request
                .query_workspace()
                .expect("query_workspace must succeed for a valid safe-API request");
            let output = query
                .evaluate()
                .expect("evaluate must succeed after Plan 17-02 executor swap");

            // The safe API returns the pair block in row-major order: out[i*nj + j].
            // Insert into the assembled matrix at the correct position.
            let pair_values = &output.tensor.owned_values;
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = pair_values[ii * nj + jj];
                }
            }

            col_offset += nj;
        }
        row_offset += ni;
    }

    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance comparison helper (Phase 15 unified: atol=1e-12, rtol=0.0)
//
// Copied verbatim from one_electron_parity.rs lines 271-292.
// ─────────────────────────────────────────────────────────────────────────────

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "output length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// AO count helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of spherical AOs for angular momentum l: 2l+1.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Number of Cartesian AOs for angular momentum l: (l+1)(l+2)/2.
fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor FFI matrix collectors (cart/sph only — vendor comparison)
//
// These helpers are ONLY compiled when CINTX_ORACLE_BUILD_VENDOR=1 is set.
// Spinor vendor FFI wrappers return complex interleaved (re/im) pairs, which
// are not directly comparable to the safe-API's real-valued Vec<f64> output
// without a complex→real projection. Spinor tests use idempotency-only below.
// ─────────────────────────────────────────────────────────────────────────────

/// Collect all shell-pair integrals for one 1e spherical operator using vendored libcint.
///
/// Returns a matrix of shape (n_ao, n_ao) packed row-major.
/// libcint 1e output is column-major (Fortran order): out[j*ni + i].
/// Converts to row-major when inserting into the result matrix.
#[cfg(has_vendor_libcint)]
fn collect_1e_sph_matrix_vendor(
    operator: &str, // "ovlp", "kin", or "nuc"
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            let _ret = match operator {
                "ovlp" => vendor_ffi::vendor_int1e_ovlp_sph(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                "kin" => vendor_ffi::vendor_int1e_kin_sph(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                "nuc" => vendor_ffi::vendor_int1e_nuc_sph(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                _ => panic!("unknown operator: {operator}"),
            };

            // libcint 1e output is column-major (Fortran order): out[j*ni + i]
            // Convert to row-major for our matrix layout
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

/// Collect all shell-pair integrals for one 1e Cartesian operator using vendored libcint.
///
/// Returns a matrix of shape (n_ao, n_ao) packed row-major.
/// libcint 1e cart output is column-major (Fortran order): out[j*ni + i].
/// Converts to row-major when inserting into the result matrix.
#[cfg(has_vendor_libcint)]
fn collect_1e_cart_matrix_vendor(
    operator: &str, // "ovlp", "kin", or "nuc"
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| ncart(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            let _ret = match operator {
                "ovlp" => vendor_ffi::vendor_int1e_ovlp_cart(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                "kin" => vendor_ffi::vendor_int1e_kin_cart(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                "nuc" => vendor_ffi::vendor_int1e_nuc_cart(
                    &mut out, &shls, atm, natm, bas, nbas, env,
                ),
                _ => panic!("unknown operator: {operator}"),
            };

            // libcint 1e output is column-major (Fortran order): out[j*ni + i]
            // Convert to row-major for our matrix layout
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

/// Collect all shell-pair 2c2e spherical integrals using vendored libcint.
///
/// Returns a matrix of shape (n_ao, n_ao) packed row-major.
/// libcint 2c2e sph output is column-major: out[j*ni + i].
/// Converts to row-major when inserting into the result matrix.
#[cfg(has_vendor_libcint)]
fn collect_2c2e_sph_matrix_vendor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            let _ret = vendor_ffi::vendor_int2c2e_sph(
                &mut out, &shls, atm, natm, bas, nbas, env,
            );

            // libcint 2c2e output is column-major: out[j*ni + i]
            // Convert to row-major for our matrix layout
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

/// Collect all shell-pair 2c2e Cartesian integrals using vendored libcint.
///
/// Returns a matrix of shape (n_ao, n_ao) packed row-major.
/// libcint 2c2e cart output is column-major: out[j*ni + i].
/// Converts to row-major when inserting into the result matrix.
#[cfg(has_vendor_libcint)]
fn collect_2c2e_cart_matrix_vendor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| ncart(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            let _ret = vendor_ffi::vendor_int2c2e_cart(
                &mut out, &shls, atm, natm, bas, nbas, env,
            );

            // libcint 2c2e output is column-major: out[j*ni + i]
            // Convert to row-major for our matrix layout
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
// Task 1: Eight cart/sph vendor-parity tests
//
// Each test is guarded #[cfg(has_vendor_libcint)] so the file still compiles
// when the vendor build is absent. The safe-API path is the system under test.
//
// Tolerance: atol=1e-12, rtol=0.0 (Phase 15 unified — D-09).
// Per D-07: 12 named #[test] functions (NOT a parametric loop) for per-symbol
// CI failure messages.
//
// OperatorId mapping (from crates/cintx-ops/src/generated/api_manifest.rs):
//   int1e_ovlp_cart=0,  int1e_ovlp_sph=1,   int1e_ovlp_spinor=2
//   int1e_kin_cart=3,   int1e_kin_sph=4,    int1e_kin_spinor=5
//   int1e_nuc_cart=6,   int1e_nuc_sph=7,    int1e_nuc_spinor=8
//   int2c2e_cart=12,    int2c2e_sph=13,     int2c2e_spinor=14
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_ovlp_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(0),
        Representation::Cart,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_cart_matrix_vendor("ovlp", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_ovlp_cart safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_ovlp_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(1),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_sph_matrix_vendor("ovlp", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_ovlp_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_kin_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(3),
        Representation::Cart,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_cart_matrix_vendor("kin", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_kin_cart safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_kin_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(4),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_sph_matrix_vendor("kin", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_kin_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_nuc_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(6),
        Representation::Cart,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_cart_matrix_vendor("nuc", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_nuc_cart safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_nuc_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(7),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_sph_matrix_vendor("nuc", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_nuc_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int2c2e_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(12),
        Representation::Cart,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_2c2e_cart_matrix_vendor(&atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int2c2e_cart safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int2c2e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(13),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_2c2e_sph_matrix_vendor(&atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int2c2e_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 2: Four spinor idempotency tests
//
// TODO(phase-18-or-later): The vendor spinor FFI wrappers in
// `crates/cintx-oracle/src/vendor_ffi.rs` return complex interleaved (re/im)
// pairs, while the safe-API's spinor output is real-valued Vec<f64>. Until a
// follow-up phase defines the complex→real projection and adds vendor-comparison
// logic, these tests use idempotency-only verification. Tracked:
// 17-RESEARCH.md Open Questions §2.
//
// These four tests do NOT carry #[cfg(has_vendor_libcint)] — they run
// unconditionally and verify determinism + at-least-one-nonzero behavior.
// ─────────────────────────────────────────────────────────────────────────────

/// Idempotency-only check used by the four spinor tests below.
///
/// No vendor-vs-safe-API comparison is performed here because the vendor spinor
/// FFI wrappers return complex interleaved (re/im) f64 pairs, while the safe
/// API returns real-valued Vec<f64>. Until a follow-up phase adds the
/// complex→real projection, spinor parity is verified by running
/// `SessionRequest::evaluate` twice and asserting byte-identical `owned_values`.
/// This catches regression to nondeterministic / zero-fill behavior even without
/// a vendor reference.
fn assert_safe_api_idempotent(operator_id: OperatorId, rep: Representation) {
    let (basis, shells) = build_h2o_sto3g_safe_basis(rep);
    let first = collect_safe_api_matrix(operator_id, rep, &basis, &shells);
    // Build basis/shells fresh for the second call to ensure there is no
    // state shared between calls (idempotency across independent sessions).
    let (basis2, shells2) = build_h2o_sto3g_safe_basis(rep);
    let second = collect_safe_api_matrix(operator_id, rep, &basis2, &shells2);
    assert_eq!(
        first.len(), second.len(),
        "spinor safe API must return the same number of elements across runs (operator={operator_id:?})"
    );
    assert_eq!(
        first, second,
        "spinor safe API must be deterministic (operator={operator_id:?}, rep={rep:?})"
    );
    let nonzero_count = first.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero_count > 0,
        "spinor safe API must produce at least one nonzero element (operator={operator_id:?}, rep={rep:?}) — zero-fill regression?"
    );
}

#[test]
fn test_int1e_ovlp_spinor_safe_api_idempotency() {
    assert_safe_api_idempotent(OperatorId::new(2), Representation::Spinor);
}

#[test]
fn test_int1e_kin_spinor_safe_api_idempotency() {
    assert_safe_api_idempotent(OperatorId::new(5), Representation::Spinor);
}

#[test]
fn test_int1e_nuc_spinor_safe_api_idempotency() {
    assert_safe_api_idempotent(OperatorId::new(8), Representation::Spinor);
}

#[test]
fn test_int2c2e_spinor_safe_api_idempotency() {
    assert_safe_api_idempotent(OperatorId::new(14), Representation::Spinor);
}
