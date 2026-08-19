//! f32 oracle gate: separate, parallel test driving `SessionQuery::evaluate_generic::<f32>()`
//! against the f64 libcint reference at empirically derived per-family tolerance floors.
//!
//! This gate is ADDITIVE (D-09) — it does NOT replace or loosen the frozen f64 byte-identity
//! gate (PREC-04). The frozen f64 gate (atol=1e-12) lives in `safe_api_arity{2,3,4}_parity.rs`
//! and is NOT touched by this file.
//!
//! Families covered: 1e (ovlp/kin/nuc, sph + cart), 2c2e (sph), 3c1e (sph), 3c2e (sph), 2e (sph).
//!
//! Tolerance model: `compare::f32_tolerance_for_family(family)` with per-family empirical rtol
//! floors derived by running the sweep, measuring `max(|f32_out as f64 - ref64| / |ref64|)` per
//! family, and setting rtol = 10× that value rounded up.
//!
//! Threat mitigation:
//! - T-20-22: floors derived empirically (not guessed); per-symbol nonzero sentinel guards against
//!   all-zero false-positive passes.
//! - T-20-23: f64 symbols NOT touched; unit tests in compare.rs assert FROZEN values.
//! - T-20-24: this is a SEPARATE file — the f64 gate files are untouched.

// Module gate: requires cpu (or rocm) feature — same as other safe-API parity tests.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_oracle::compare::f32_tolerance_for_family;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G safe-API fixture (PTR_ENV_START-aware, identical to arity2/3/4 files)
//
// CRITICAL: libcint reserves env[0..PTR_ENV_START] for global parameters.
// Numeric values must be byte-identical to the f64 parity tests so the f32 and f64
// gate results are directly comparable.
// ─────────────────────────────────────────────────────────────────────────────

const N_SHELLS: usize = 5;

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o  = Atom::try_new(8, [0.0, 0.0, 0.0],        NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078],  NuclearModel::Point, None, None).unwrap();
    let atom_h2 = Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G exponents/coefficients (Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657, 1969).
    // Identical values to safe_api_arity2_parity.rs so vendor f64 reference is the same.

    let shell_o1s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    let shell_o2s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[-0.09996723, 0.39951283, 0.70011547]),
    ).unwrap());

    let shell_o2p = Arc::new(Shell::try_new(
        0, 1, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[0.15591627, 0.60768372, 0.39195739]),
    ).unwrap());

    let shell_h1_1s = Arc::new(Shell::try_new(
        1, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

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
// f32 safe-API matrix collector (arity-2)
//
// Mirrors `collect_safe_api_matrix` from safe_api_arity2_parity.rs but calls
// `evaluate_generic::<f32>()` and returns Vec<f32> for the f32 values.
// Callers cast each element `as f64` before differencing against the f64 reference.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_matrix_f32(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
) -> Vec<f32> {
    let shell_nao: Vec<usize> = shells.iter().map(|s| s.ao_per_shell()).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f32; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];
            let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple for 2-shell pair");
            let request = SessionRequest::new(
                operator_id,
                rep,
                basis,
                shell_tuple,
                ExecutionOptions::default(),
            );
            let query = request
                .query_workspace()
                .expect("query_workspace must succeed");
            let output = query
                .evaluate_generic::<f32>()
                .expect("evaluate_generic::<f32>() must succeed");

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
// f32 safe-API tuple buffer collector (arity-3)
//
// Returns Vec<f32> for one 3-shell tuple. Caller casts as f64 for comparison.
// Direct buffer compare (no transpose needed) — same as arity-3 f64 gate.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_tuple_buffer_f32(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    tuple_shells: &[Arc<Shell>],
) -> Vec<f32> {
    let shell_tuple = ShellTuple::try_from_iter(tuple_shells.iter().cloned())
        .expect("tuple within SHELL_TUPLE_CAPACITY=4");
    let request = SessionRequest::new(
        operator_id,
        rep,
        basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request
        .query_workspace()
        .expect("query_workspace must succeed for arity-3 f32 request");
    let output = query
        .evaluate_generic::<f32>()
        .expect("evaluate_generic::<f32>() must succeed for arity-3");
    output.tensor.owned_values
}

// ─────────────────────────────────────────────────────────────────────────────
// f32 safe-API tuple buffer collector (arity-4)
//
// Returns Vec<f32> for one 4-shell tuple.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_tuple_buffer_f32_arity4(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    tuple_shells: &[Arc<Shell>],
) -> Vec<f32> {
    let shell_tuple = ShellTuple::try_from_iter(tuple_shells.iter().cloned())
        .expect("tuple within SHELL_TUPLE_CAPACITY=4");
    let request = SessionRequest::new(
        operator_id,
        rep,
        basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request
        .query_workspace()
        .expect("query_workspace must succeed for arity-4 f32 request");
    let output = query
        .evaluate_generic::<f32>()
        .expect("evaluate_generic::<f32>() must succeed for arity-4");
    output.tensor.owned_values
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance comparison helper for f32 oracle gate.
//
// `reference` is the f64 libcint output.
// `observed` is the f32 output cast to f64 element-wise.
//
// Returns (mismatch_count, max_rel_error) for empirical floor derivation.
// ─────────────────────────────────────────────────────────────────────────────

fn count_mismatches_f32(
    reference: &[f64],
    observed_f32: &[f32],
    atol: f64,
    rtol: f64,
    zero_threshold: f64,
) -> (usize, f64) {
    assert_eq!(
        reference.len(),
        observed_f32.len(),
        "output length mismatch: f64_ref={} f32_obs={}",
        reference.len(),
        observed_f32.len(),
    );
    let mut mismatches = 0usize;
    let mut max_rel_error = 0.0_f64;
    for (i, (&ref_val, &obs_f32)) in reference.iter().zip(observed_f32.iter()).enumerate() {
        let obs_val = obs_f32 as f64;
        let abs_error = (obs_val - ref_val).abs();
        let abs_ref = ref_val.abs();

        // Compute relative error for diagnostic recording (empirical floor).
        if abs_ref > zero_threshold {
            let rel_err = abs_error / abs_ref;
            if rel_err > max_rel_error {
                max_rel_error = rel_err;
            }
        }

        // Tolerance check matching diff_summary logic in compare.rs.
        let passed = if abs_ref < zero_threshold {
            abs_error <= atol
        } else {
            abs_error <= atol + rtol * abs_ref
        };
        if !passed {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "  F32 MISMATCH at index {i}: ref64={ref_val:.15e}, obs_f32={obs_f32:.8e}, \
                     obs_f64={obs_val:.15e}, abs_err={abs_error:.3e}, rtol_thr={:.3e}",
                    atol + rtol * abs_ref
                );
            }
        }
    }
    (mismatches, max_rel_error)
}

// ─────────────────────────────────────────────────────────────────────────────
// AO count helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of spherical AOs for angular momentum l: 2l+1.
fn nsph(l: i32) -> usize { (2 * l + 1) as usize }
/// Number of Cartesian AOs for angular momentum l: (l+1)(l+2)/2.
fn ncart(l: i32) -> usize { ((l + 1) * (l + 2) / 2) as usize }

// ─────────────────────────────────────────────────────────────────────────────
// Vendor f64 reference collectors (f64 only — reference side is FROZEN)
//
// These collect the libcint f64 reference matrices, identical to those used
// by the frozen f64 gate in safe_api_arity2_parity.rs. The reference side is
// never f32; we compare `f32_out as f64` against the f64 reference.
// ─────────────────────────────────────────────────────────────────────────────

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};

fn build_h2o_sto3g_raw() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    let mut env = vec![0.0_f64; PTR_ENV_START];
    let o_coord_ptr = env.len() as i32;   env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;  env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;  env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32;      env.push(0.0);
    let o1s_exp_ptr = env.len() as i32;   env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; env.extend_from_slice(&o_1s_coeff);
    let o2s_exp_ptr = env.len() as i32;   env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; env.extend_from_slice(&o_2s_coeff);
    let o2p_exp_ptr = env.len() as i32;   env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; env.extend_from_slice(&o_2p_coeff);
    let h1s_exp_ptr = env.len() as i32;   env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; env.extend_from_slice(&h_1s_coeff);

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
    bas[0 * BAS_SLOTS + ATOM_OF] = 0; bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3; bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr; bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 0; bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3; bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr; bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    bas[2 * BAS_SLOTS + ATOM_OF] = 0; bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3; bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr; bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    bas[3 * BAS_SLOTS + ATOM_OF] = 1; bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3; bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr; bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    bas[4 * BAS_SLOTS + ATOM_OF] = 2; bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3; bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr; bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor reference collector: 1e sph matrix (f64 reference side, FROZEN)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
fn collect_1e_sph_matrix_vendor_ref(
    operator: &str,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
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
                "ovlp" => vendor_ffi::vendor_int1e_ovlp_sph(&mut out, &shls, atm, natm, bas, nbas, env),
                "kin"  => vendor_ffi::vendor_int1e_kin_sph( &mut out, &shls, atm, natm, bas, nbas, env),
                "nuc"  => vendor_ffi::vendor_int1e_nuc_sph( &mut out, &shls, atm, natm, bas, nbas, env),
                _ => panic!("unknown operator: {operator}"),
            };
            // vendor 1e is column-major; convert to row-major
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

#[cfg(has_vendor_libcint)]
fn collect_1e_cart_matrix_vendor_ref(
    operator: &str,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
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
                "ovlp" => vendor_ffi::vendor_int1e_ovlp_cart(&mut out, &shls, atm, natm, bas, nbas, env),
                "kin"  => vendor_ffi::vendor_int1e_kin_cart( &mut out, &shls, atm, natm, bas, nbas, env),
                "nuc"  => vendor_ffi::vendor_int1e_nuc_cart( &mut out, &shls, atm, natm, bas, nbas, env),
                _ => panic!("unknown operator: {operator}"),
            };
            // vendor 1e cart is column-major; convert to row-major
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

#[cfg(has_vendor_libcint)]
fn collect_2c2e_sph_matrix_vendor_ref(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
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
            let _ret = vendor_ffi::vendor_int2c2e_sph(&mut out, &shls, atm, natm, bas, nbas, env);
            // vendor 2c2e is column-major; convert to row-major
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
// Macro-helper: measure max_rel_error across all shell tuples for one family/operator.
// Prints the empirical floor result to stderr.
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate max relative error across all arity-2 shell pairs.
fn measure_max_rel_error_arity2(
    f32_matrix: &[f32],
    f64_ref: &[f64],
    zero_threshold: f64,
) -> f64 {
    assert_eq!(f32_matrix.len(), f64_ref.len());
    let mut max_rel = 0.0_f64;
    for (&r, &o) in f64_ref.iter().zip(f32_matrix.iter()) {
        let abs_ref = r.abs();
        if abs_ref > zero_threshold {
            let rel = (o as f64 - r).abs() / abs_ref;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    max_rel
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 1e (sph) ────────────────────────────────────────────────────────
//
// OperatorId mapping (same as safe_api_arity2_parity.rs):
//   int1e_ovlp_sph=1, int1e_kin_sph=4, int1e_nuc_sph=7
//
// Phase 18 convention: nonzero sentinel asserted for every test.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_ovlp_sph_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(1),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_sph_matrix_vendor_ref("ovlp", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    // Anti-zero-fill sentinel (Phase 18 convention)
    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 ovlp_sph: all-zero fill detected — f32 kernel regression?");

    // Empirical max_rel_error diagnostic
    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/ovlp_sph: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_ovlp_sph: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_kin_sph_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(4),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_sph_matrix_vendor_ref("kin", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 kin_sph: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/kin_sph: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_kin_sph: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_nuc_sph_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(7),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_sph_matrix_vendor_ref("nuc", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 nuc_sph: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/nuc_sph: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_nuc_sph: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 1e (cart) ───────────────────────────────────────────────────────
//
// OperatorId: int1e_ovlp_cart=0, int1e_kin_cart=3, int1e_nuc_cart=6
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_ovlp_cart_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(0),
        Representation::Cart,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_cart_matrix_vendor_ref("ovlp", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 ovlp_cart: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/ovlp_cart: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_ovlp_cart: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_kin_cart_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(3),
        Representation::Cart,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_cart_matrix_vendor_ref("kin", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 kin_cart: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/kin_cart: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_kin_cart: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int1e_nuc_cart_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(6),
        Representation::Cart,
        &basis,
        &shells,
    );
    let f64_ref = collect_1e_cart_matrix_vendor_ref("nuc", &atm, &bas, &env);
    let tol = f32_tolerance_for_family("1e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 nuc_cart: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 1e/nuc_cart: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int1e_nuc_cart: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 2c2e (sph) ──────────────────────────────────────────────────────
//
// OperatorId: int2c2e_sph=13
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int2c2e_sph_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let f32_matrix = collect_safe_api_matrix_f32(
        OperatorId::new(13),
        Representation::Spheric,
        &basis,
        &shells,
    );
    let f64_ref = collect_2c2e_sph_matrix_vendor_ref(&atm, &bas, &env);
    let tol = f32_tolerance_for_family("2c2e");

    let nonzero = f32_matrix.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 2c2e_sph: all-zero fill detected");

    let max_rel = measure_max_rel_error_arity2(&f32_matrix, &f64_ref, tol.zero_threshold);
    eprintln!("f32 2c2e/sph: max_rel_error={max_rel:.3e} (floor={:.3e})", tol.rtol);

    let (mismatches, _) = count_mismatches_f32(&f64_ref, &f32_matrix, tol.atol, tol.rtol, tol.zero_threshold);
    assert_eq!(
        mismatches, 0,
        "f32 int2c2e_sph: {mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 3c1e (sph) ──────────────────────────────────────────────────────
//
// OperatorId: int3c1e_sph=18 (from safe_api_arity3_parity.rs:306)
// Direct buffer compare — arity-3 kernels write F-order matching vendor output.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int3c1e_sph_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let tol = f32_tolerance_for_family("3c1e");

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut max_rel_family = 0.0_f64;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                let ni = shell_nao[i];
                let nj = shell_nao[j];
                let nk = shell_nao[k];
                let n_elem = ni * nj * nk;

                let f32_out = collect_safe_api_tuple_buffer_f32(
                    OperatorId::new(18),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut f64_ref = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                vendor_ffi::vendor_int3c1e_sph(&mut f64_ref, &shls, &atm, natm, &bas, nbas, &env);

                let nonzero = f32_out.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
                if nonzero > 0 { any_nonzero = true; }

                let (mc, mr) = count_mismatches_f32(&f64_ref, &f32_out, tol.atol, tol.rtol, tol.zero_threshold);
                total_mismatches += mc;
                if mr > max_rel_family { max_rel_family = mr; }
            }
        }
    }

    eprintln!("f32 3c1e/sph: max_rel_error={max_rel_family:.3e} (floor={:.3e})", tol.rtol);
    assert!(any_nonzero, "f32 3c1e_sph: all-zero fill detected across all triples");
    assert_eq!(
        total_mismatches, 0,
        "f32 int3c1e_sph: {total_mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 3c2e (sph) ──────────────────────────────────────────────────────
//
// OperatorId: int3c2e_ip1_sph=20 (cintx computes plain 3c2e; vendor ref is int3c2e_sph
// per safe_api_arity3_parity.rs NOTE: kernel-misnomer disposition).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int3c2e_sph_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let tol = f32_tolerance_for_family("3c2e");

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut max_rel_family = 0.0_f64;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                let ni = shell_nao[i];
                let nj = shell_nao[j];
                let nk = shell_nao[k];
                let n_elem = ni * nj * nk;

                // OperatorId 23 = int3c2e_sph
                let f32_out = collect_safe_api_tuple_buffer_f32(
                    OperatorId::new(23),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut f64_ref = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                // Reference is plain int3c2e_sph (matching the kernel-misnomer disposition)
                vendor_ffi::vendor_int3c2e_sph(&mut f64_ref, &shls, &atm, natm, &bas, nbas, &env);

                let nonzero = f32_out.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
                if nonzero > 0 { any_nonzero = true; }

                let (mc, mr) = count_mismatches_f32(&f64_ref, &f32_out, tol.atol, tol.rtol, tol.zero_threshold);
                total_mismatches += mc;
                if mr > max_rel_family { max_rel_family = mr; }
            }
        }
    }

    eprintln!("f32 3c2e/sph: max_rel_error={max_rel_family:.3e} (floor={:.3e})", tol.rtol);
    assert!(any_nonzero, "f32 3c2e_sph: all-zero fill detected across all triples");
    assert_eq!(
        total_mismatches, 0,
        "f32 int3c2e_sph: {total_mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: 2e (sph) ────────────────────────────────────────────────────────
//
// OperatorId: int2e_sph=10 (from safe_api_arity4_parity.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_f32_int2e_sph_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_raw();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let tol = f32_tolerance_for_family("2e");

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut max_rel_family = 0.0_f64;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shell_nao[i];
                    let nj = shell_nao[j];
                    let nk = shell_nao[k];
                    let nl = shell_nao[l];
                    let n_elem = ni * nj * nk * nl;

                    let f32_out = collect_safe_api_tuple_buffer_f32_arity4(
                        OperatorId::new(10),
                        Representation::Spheric,
                        &basis,
                        &[
                            shells[i].clone(),
                            shells[j].clone(),
                            shells[k].clone(),
                            shells[l].clone(),
                        ],
                    );

                    let mut f64_ref = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    vendor_ffi::vendor_int2e_sph(&mut f64_ref, &shls, &atm, natm, &bas, nbas, &env);

                    let nonzero = f32_out.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
                    if nonzero > 0 { any_nonzero = true; }

                    let (mc, mr) = count_mismatches_f32(&f64_ref, &f32_out, tol.atol, tol.rtol, tol.zero_threshold);
                    total_mismatches += mc;
                    if mr > max_rel_family { max_rel_family = mr; }
                }
            }
        }
    }

    eprintln!("f32 2e/sph: max_rel_error={max_rel_family:.3e} (floor={:.3e})", tol.rtol);
    assert!(any_nonzero, "f32 2e_sph: all-zero fill detected across all quartets");
    assert_eq!(
        total_mismatches, 0,
        "f32 int2e_sph: {total_mismatches} elements exceed f32 floor atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Unconditional smoke test: nonzero + finite output from evaluate_generic::<f32>()
//
// Runs even without the vendor build — verifies f32 evaluate_generic plumbing
// is not zero-filling or producing NaN/Inf. Mirrors the Phase 18 sentinel pattern.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_f32_evaluate_generic_produces_nonzero_finite_output() {
    // Drive int1e_ovlp_sph (OperatorId=1) with O 1s/O 2s shell pair.
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    // Single O1s–O2s pair (both s-shells; guaranteed nonzero overlap)
    let shell_tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()])
        .expect("ShellTuple for 2-shell pair");
    let request = SessionRequest::new(
        OperatorId::new(1),
        Representation::Spheric,
        &basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request
        .query_workspace()
        .expect("query_workspace for f32 smoke test");
    let output = query
        .evaluate_generic::<f32>()
        .expect("evaluate_generic::<f32>() smoke test");

    let values = &output.tensor.owned_values;
    assert!(!values.is_empty(), "f32 evaluate_generic returned empty output");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "f32 evaluate_generic returned non-finite value"
    );
    let nonzero = values.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
    assert!(nonzero > 0, "f32 evaluate_generic returned all-zero output for O1s–O2s overlap");
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── FAMILY: f12 (derivative, ncomp=3) — GAP 2 CLOSURE (PREC-05, Plan 20-11) ─
//
// operator: int2e_stg_ip1_sph (OperatorId=107, ncomp=3: gradient on electron 1)
//
// CR-01/CR-02 LOAD-BEARING CONTEXT:
//   With ncomp=3, the planner sets staging_elements = base_elements * 3.
//   After bytemuck::cast_slice_mut, the f32 slice has chunk_len * 2 lanes.
//   Pre-20-10 code derived copy_len/not0 from staging.len() (== chunk_len*2),
//   not from the true output element count. For ncomp=3:
//     staging_elements = base * 3 > chunk_len
//   so staging.len() inside the typed inner (after bytemuck) == chunk_len*2,
//   which is LARGER than the true output element count AND larger than the
//   actual f32 output region. The pre-fix code read/wrote stale upper-half
//   lanes, returning silently wrong f32 values. These tests FAIL pre-20-10
//   and PASS after (CR-01 + CR-02 fixes in commit 5ba79fb).
//
// Gate pattern: #[cfg(all(has_vendor_libcint, feature = "with-f12"))]
//   - Drives evaluate_generic::<f32>() and reads owned_values: Vec<f32>
//   - Compares (f32_out as f64) against vendor_int2e_stg_ip1_sph (f64 libcint ref)
//   - Uses f32_tolerance_for_family("f12") at the empirically derived floor
//   - Anti-zero-fill sentinel + finite sentinel guard false-positive passes
// ─────────────────────────────────────────────────────────────────────────────

/// Build H2O STO-3G safe-API basis for f12 tests.
///
/// Mirrors build_h2o_sto3g_safe_basis (Spheric) but returns the shells as a flat
/// Vec for arity-4 quartet iteration. Molecule and numerical values are identical
/// to the raw build so vendor reference and safe-API f32 outputs are comparable.
#[cfg(feature = "with-f12")]
fn build_h2o_sto3g_f12_safe_basis() -> (cintx_core::BasisSet, Vec<Arc<cintx_core::Shell>>) {
    build_h2o_sto3g_safe_basis(Representation::Spheric)
}

/// Collect int2e_stg_ip1_sph (ncomp=3) f32 output via the safe evaluate_generic API.
///
/// Drives `SessionRequest::evaluate_generic::<f32>()` with zeta=1.2 and
/// returns `output.tensor.owned_values` (Vec<f32>, length = 3 * ni * nj * nk * nl).
/// This is the exact entry point that the CR-01/CR-02 corruption silently broke
/// for ncomp=3 on the pre-20-10 code.
#[cfg(feature = "with-f12")]
fn collect_stg_ip1_f32(
    basis: &cintx_core::BasisSet,
    tuple_shells: &[Arc<cintx_core::Shell>],
) -> Vec<f32> {
    use cintx_runtime::ExecutionOptions;
    // OperatorId 107 = int2e_stg_ip1_sph (int2e_stg_sph is 106; ip1 is next in manifest).
    const STG_IP1_SPH_OPERATOR_ID: u32 = 107;
    let shell_tuple = ShellTuple::try_from_iter(tuple_shells.iter().cloned())
        .expect("ShellTuple for 4-shell f12 ip1 quartet");
    let request = SessionRequest::new(
        OperatorId::new(STG_IP1_SPH_OPERATOR_ID),
        Representation::Spheric,
        basis,
        shell_tuple,
        ExecutionOptions {
            f12_zeta: Some(1.2_f64),
            ..ExecutionOptions::default()
        },
    );
    let query = request
        .query_workspace()
        .expect("query_workspace must succeed for f12 ip1 f32 request");
    let output = query
        .evaluate_generic::<f32>()
        .expect("evaluate_generic::<f32>() must succeed for int2e_stg_ip1_sph");
    output.tensor.owned_values
}

/// f32 oracle parity gate for int2e_stg_ip1_sph (ncomp=3) — the CR-01/CR-02 corruption regime.
///
/// This test is LOAD-BEARING (Plan 20-11 / PREC-05 Gap 2 closure):
///   - ncomp=3 => staging_elements = base*3, which is > chunk_len on the f32 path.
///   - Pre-20-10 code derived copy_len/not0 from staging.len() (doubled f32 lane count),
///     silently returning wrong f32 values for all ncomp>1 operators.
///   - This test FAILS against pre-20-10 code and PASSES after (commit 5ba79fb).
///
/// Quartets: uses the same 3 quartets as the f64 oracle (oracle_parity_int2e_stg_ip1_sph):
///   SHLS_4_SS=[0,1,0,1], SHLS_4_HH=[3,4,3,4], SHLS_4_SP=[0,2,0,2]
/// These are the known-good quartets verified at f64 oracle quality; asymmetric
/// quartets with l>0 shells (e.g. (0,2,2,2)) expose a pre-existing f12 kernel
/// bug for non-symmetric shell orderings that is OUT OF SCOPE for this plan
/// (deferred — separate from the CR-01/CR-02 f32 length-contract bug).
///
/// CR-01/CR-02 regime confirmation:
///   SHLS_4_SS: base=1, ncomp=3 → staging_elements=3, pre-fix staging_f32.len()=6 > 3.
///   SHLS_4_HH: base=1, ncomp=3 → same.
///   SHLS_4_SP: base=9, ncomp=3 → staging_elements=27, pre-fix staging_f32.len()=54 > 27.
/// All three quartets exercise staging_elements > chunk_len (the CR-01/CR-02 corruption case).
///
/// Verification:
///   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12 \
///     --test f32_parity test_f32_int2e_stg_ip1_sph_parity
#[test]
#[cfg(all(has_vendor_libcint, feature = "with-f12"))]
fn test_f32_int2e_stg_ip1_sph_parity() {
    use cintx_oracle::vendor_ffi;
    use cintx_oracle::fixtures::build_h2o_sto3g_f12;

    // Build the raw fixture for the vendor reference (f64 libcint, zeta=1.2).
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2_f64);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    // Build the safe-API basis for cintx f32 evaluation.
    let (basis, shells) = build_h2o_sto3g_f12_safe_basis();

    // ncomp = 3: d/dx, d/dy, d/dz of STG on electron 1.
    // staging_elements = ni*nj*nk*nl * 3 > chunk_len — the CR-01/CR-02 regime.
    let ncomp = 3usize;

    // Tolerance: f32 family floor for "f12" (empirically derived — see SUMMARY).
    let tol = f32_tolerance_for_family("f12");

    // Shell quartets matching the f64 oracle (oracle_parity_int2e_stg_ip1_sph):
    //   [0,1,0,1] = O-1s, O-2s, O-1s, O-2s  (all s; base=1*1*1*1=1, staging=3)
    //   [3,4,3,4] = H1-1s, H2-1s, H1-1s, H2-1s (all s; base=1, staging=3)
    //   [0,2,0,2] = O-1s, O-2p, O-1s, O-2p  (mixed s/p; base=1*3*1*3=9, staging=27)
    // All three have staging_elements = ncomp * base > chunk_len for the f32 path —
    // the exact CR-01/CR-02 corruption regime.
    // Note: shells[2]=O-2p has l=1, 3 spherical AOs — this is the multi-component case.
    let shell_quartets: &[[usize; 4]] = &[
        [0, 1, 0, 1], // SHLS_4_SS equivalent
        [3, 4, 3, 4], // SHLS_4_HH equivalent
        [0, 2, 0, 2], // SHLS_4_SP equivalent: exercises ncomp=3 with p-shell (staging=27)
    ];

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao_sph: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut max_rel_family = 0.0_f64;

    for &[i, j, k, l] in shell_quartets {
        let ni = shell_nao_sph[i];
        let nj = shell_nao_sph[j];
        let nk = shell_nao_sph[k];
        let nl = shell_nao_sph[l];
        // ncomp * (ni*nj*nk*nl) is the true output element count.
        let n_elem = ncomp * ni * nj * nk * nl;

        // ── cintx f32 via the safe API (evaluate_generic::<f32>()) ──────────
        // This drives the f12 kernel through the F32 arm (CR-01/CR-02 regime).
        let f32_out = collect_stg_ip1_f32(
            &basis,
            &[shells[i].clone(), shells[j].clone(), shells[k].clone(), shells[l].clone()],
        );

        assert_eq!(
            f32_out.len(), n_elem,
            "int2e_stg_ip1_sph f32 output length mismatch: got {} expected {} for ({},{},{},{})",
            f32_out.len(), n_elem, i, j, k, l
        );

        // ── f64 libcint vendor reference (same quartets as f64 oracle) ───────
        let shls = [i as i32, j as i32, k as i32, l as i32];
        let mut vendor_out = vec![0.0_f64; n_elem];
        vendor_ffi::vendor_int2e_stg_ip1_sph(
            &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
        );

        // ── anti-zero-fill sentinel ──────────────────────────────────────────
        let nonzero_count = f32_out.iter().filter(|&&v| v.abs() > 1e-18_f32).count();
        if nonzero_count > 0 {
            any_nonzero = true;
        }

        // ── finite sentinel ──────────────────────────────────────────────────
        assert!(
            f32_out.iter().all(|v| v.is_finite()),
            "int2e_stg_ip1_sph f32 output non-finite for shls ({},{},{},{})",
            i, j, k, l
        );

        // ── max_rel diagnostic (records the empirical floor) ─────────────────
        let (mc, mr) = count_mismatches_f32(
            &vendor_out, &f32_out, tol.atol, tol.rtol, tol.zero_threshold,
        );
        if mr > max_rel_family {
            max_rel_family = mr;
        }
        total_mismatches += mc;
        if mc > 0 {
            eprintln!(
                "int2e_stg_ip1_sph f32 FAIL: {mc} mismatches for shls ({},{},{},{}) \
                 [CR-01/CR-02 regime: ncomp=3 => staging_elements=base*3 > chunk_len, max_rel={mr:.3e}]",
                i, j, k, l
            );
        } else {
            eprintln!(
                "int2e_stg_ip1_sph f32 PASS: shls ({},{},{},{}) max_rel={mr:.3e} (floor={:.3e})",
                i, j, k, l, tol.rtol
            );
        }
    }

    eprintln!(
        "f32 f12/stg_ip1_sph (ncomp=3): max_rel_error={max_rel_family:.3e} (floor={:.3e})",
        tol.rtol
    );
    assert!(
        any_nonzero,
        "int2e_stg_ip1_sph f32 output all-zero across all tested quartets (anti-zero-fill sentinel)"
    );
    assert_eq!(
        total_mismatches, 0,
        "int2e_stg_ip1_sph f32 parity FAIL: {total_mismatches} mismatches \
         (CR-01/CR-02 regime ncomp=3) — atol={:.0e}/rtol={:.0e}",
        tol.atol, tol.rtol
    );
}
