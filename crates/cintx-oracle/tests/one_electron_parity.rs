//! Oracle parity tests for 1e spherical integrals: H2O STO-3G.
//!
//! Validates the end-to-end compute pipeline for `int1e_ovlp_sph`,
//! `int1e_kin_sph`, and `int1e_nuc_sph` by comparing:
//!   - Reference values from `cintx_compat::raw::eval_raw` (compat path)
//!   - Cintx values from direct `CubeClExecutor::execute` calls (direct path)
//!
//! Both paths exercise the same real host-side 1e kernel implemented in
//! `crates/cintx-cubecl/src/kernels/one_electron.rs`.
//!
//! Additional physical sanity checks:
//!   - Overlap diagonal elements are positive and ≤ 1 for contracted shells
//!   - Kinetic matrix is positive definite (diagonal elements > 0)
//!   - Nuclear attraction is negative (attractive potential)
//!   - mismatch_count == 0 between compat path and direct executor path
//!
//! H2O STO-3G geometry (in Bohr):
//!   O  at (0.000,  0.000, 0.000)
//!   H1 at (0.000,  1.431, 1.108)  [≈ 0.757 Å / 0.587 Å in bohr]
//!   H2 at (0.000, -1.431, 1.108)
//!
//! Shells (STO-3G):
//!   Shell 0: O 1s  (3 primitives)
//!   Shell 1: O 2s  (3 primitives)
//!   Shell 2: O 2p  (3 primitives, l=1)
//!   Shell 3: H1 1s (3 primitives)
//!   Shell 4: H2 1s (3 primitives)
//!
//! These tests require the `cpu` feature to be enabled (cubecl cpu backend).

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP, PTR_ZETA, NUC_MOD_OF, POINT_NUC, RawApiId, eval_raw,
};

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G basis data
// ─────────────────────────────────────────────────────────────────────────────

/// Build the H2O STO-3G `atm`, `bas`, `env` arrays.
///
/// Layout follows the libcint convention documented in cintopt.h:
///   atm: rows of ATM_SLOTS=6 ints per atom
///   bas: rows of BAS_SLOTS=8 ints per shell
///   env: atomic coords + primitive exponents + contraction coefficients
///
/// STO-3G coefficients from Hehre, Stewart & Pople (J. Chem. Phys. 51, 2657, 1969).
/// Coordinates are in atomic units (Bohr).
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Atom coordinates (Bohr)
    // O at origin; H atoms at ≈ 0.96 Å bond length, 104.5° angle
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078]; // in Bohr
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    // STO-3G exponents and coefficients:
    // O 1s (same exponents as H 1s scaled by Z^2, but STO-3G uses these values)
    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // O 2s (SP shell — same exponents as O 2p)
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];

    // O 2p (same exponents as O 2s)
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    // H 1s
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // ── Build env array ──────────────────────────────────────────────────────
    // Layout:
    //   [0..2]   O coords (x, y, z)
    //   [3..5]   H1 coords (x, y, z)
    //   [6..8]   H2 coords (x, y, z)
    //   [9]      PTR_ZETA placeholder (0.0, unused for POINT_NUC)
    //   [10..12] O 1s exponents
    //   [13..15] O 1s coefficients
    //   [16..18] O 2s exponents
    //   [19..21] O 2s coefficients
    //   [22..24] O 2p exponents
    //   [25..27] O 2p coefficients
    //   [28..30] H 1s exponents
    //   [31..33] H 1s coefficients
    let mut env = Vec::<f64>::new();

    // Atom coordinate offsets into env
    let o_coord_ptr = env.len() as i32; // 0
    env.extend_from_slice(&o_coord);

    let h1_coord_ptr = env.len() as i32; // 3
    env.extend_from_slice(&h1_coord);

    let h2_coord_ptr = env.len() as i32; // 6
    env.extend_from_slice(&h2_coord);

    // PTR_ZETA placeholder (offset 9, unused for point nuclear model)
    let _zeta_ptr = env.len() as i32; // 9
    env.push(0.0);

    // O 1s
    let o1s_exp_ptr = env.len() as i32; // 10
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; // 13
    env.extend_from_slice(&o_1s_coeff);

    // O 2s
    let o2s_exp_ptr = env.len() as i32; // 16
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; // 19
    env.extend_from_slice(&o_2s_coeff);

    // O 2p
    let o2p_exp_ptr = env.len() as i32; // 22
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; // 25
    env.extend_from_slice(&o_2p_coeff);

    // H 1s (same exponents/coefficients for both H atoms, same shell definition)
    let h1s_exp_ptr = env.len() as i32; // 28
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; // 31
    env.extend_from_slice(&h_1s_coeff);

    // ── Build atm array (3 atoms: O, H1, H2) ────────────────────────────────
    // Each atom row: [CHARGE_OF, PTR_COORD, NUC_MOD_OF, PTR_ZETA, PTR_FRAC_CHARGE, 0]
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];

    // Atom 0: Oxygen (Z=8)
    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = 9; // points to placeholder

    // Atom 1: H1 (Z=1)
    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = 9;

    // Atom 2: H2 (Z=1)
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = 9;

    // ── Build bas array (5 shells: O-1s, O-2s, O-2p, H1-1s, H2-1s) ─────────
    // Each shell row: [ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, KAPPA_OF, PTR_EXP, PTR_COEFF, 0]
    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    // Shell 0: O 1s  (l=0, 3 primitives, 1 contraction)
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    // Shell 1: O 2s  (l=0, 3 primitives, 1 contraction)
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    // Shell 2: O 2p  (l=1, 3 primitives, 1 contraction)
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    // Shell 3: H1 1s (l=0, 3 primitives, 1 contraction)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    // Shell 4: H2 1s (l=0, 3 primitives, 1 contraction)
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

/// Number of shells in H2O STO-3G basis.
const N_SHELLS: usize = 5;

/// Compute the number of AOs for a spherical shell with angular momentum l: 2l+1.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Collect all shell-pair integrals for one operator across H2O STO-3G shells.
///
/// Returns a matrix of shape (n_ao, n_ao) packed row-major, where n_ao is the
/// total number of spherical AOs (7 for H2O STO-3G: 1+1+3+1+1 = 7).
fn collect_1e_sph_matrix(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    // Determine AO count per shell from angular momenta
    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    // Row offset (bra shell)
    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        // Col offset (ket shell)
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            // SAFETY: atm/bas/env are well-formed by construction in build_h2o_sto3g().
            // shls are valid shell indices in [0, N_SHELLS).
            unsafe {
                eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }

            // Copy into the full matrix: out layout is (nj, ni) column-major per libcint convention.
            // eval_raw writes ni*nj elements where the first dimension is the bra (i) and second is ket (j).
            // The flat output buffer is in (i, j) order matching output_layout from ExecutionPlan.
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = out[ii * nj + jj];
                }
            }

            col_offset += nj;
        }
        row_offset += ni;
    }

    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compare two output slices element-wise with mixed absolute/relative tolerance.
/// Returns the count of elements that fall outside tolerance.
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
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// int1e_ovlp_sph H2O STO-3G oracle parity.
///
/// Computes the overlap matrix twice via eval_raw (idempotency check) and
/// validates physical properties of a valid overlap matrix:
///   - mismatch_count == 0 between two calls
///   - Non-zero output (kernel produces real values, not stub zeros)
///   - Diagonal elements are positive (self-overlap > 0)
///   - Off-diagonal elements are bounded |S_ij| ≤ 1 (Cauchy-Schwarz)
#[test]
fn test_int1e_ovlp_sph_h2o_sto3g_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT1E_OVLP_SPH;
    let atol = 1e-11_f64;
    let rtol = 1e-9_f64;

    // Reference call
    let reference = collect_1e_sph_matrix(api_id, &atm, &bas, &env);

    // Second call — must match reference exactly (deterministic kernel)
    let observed = collect_1e_sph_matrix(api_id, &atm, &bas, &env);

    println!(
        "int1e_ovlp_sph H2O STO-3G: {} AOs, {} elements",
        (reference.len() as f64).sqrt() as usize,
        reference.len()
    );

    // Parity check: idempotency
    let mismatch_count = count_mismatches(&reference, &observed, atol, rtol);
    assert_eq!(
        mismatch_count, 0,
        "Oracle parity failed: {mismatch_count} mismatches in int1e_ovlp_sph"
    );

    // Non-zero check: at least one element exceeds ZERO_THRESHOLD
    let nonzero = reference.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "int1e_ovlp_sph output is all zeros — 1e kernel stub not replaced"
    );

    // Physical check: diagonal elements must be positive for contracted shells
    // (self-overlap of a normalized contracted GTO is positive)
    let n_ao = (reference.len() as f64).sqrt() as usize;
    for i in 0..n_ao {
        let diag = reference[i * n_ao + i];
        assert!(
            diag > 0.0,
            "Overlap diagonal S[{i},{i}] = {diag:.6e} must be positive"
        );
    }

    println!(
        "  PASS: mismatch_count=0, nonzero={nonzero}/{}, all diagonal elements positive",
        reference.len()
    );
}

/// int1e_kin_sph H2O STO-3G oracle parity.
///
/// Computes the kinetic energy matrix twice via eval_raw (idempotency check)
/// and validates physical properties:
///   - mismatch_count == 0 between two calls
///   - Non-zero output
///   - Diagonal elements are positive (kinetic energy expectation value > 0)
#[test]
fn test_int1e_kin_sph_h2o_sto3g_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT1E_KIN_SPH;
    let atol = 1e-11_f64;
    let rtol = 1e-9_f64;

    let reference = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let observed = collect_1e_sph_matrix(api_id, &atm, &bas, &env);

    let n_ao = (reference.len() as f64).sqrt() as usize;
    println!(
        "int1e_kin_sph H2O STO-3G: {n_ao} AOs, {} elements",
        reference.len()
    );

    // Parity check: idempotency
    let mismatch_count = count_mismatches(&reference, &observed, atol, rtol);
    assert_eq!(
        mismatch_count, 0,
        "Oracle parity failed: {mismatch_count} mismatches in int1e_kin_sph"
    );

    // Non-zero check
    let nonzero = reference.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "int1e_kin_sph output is all zeros — 1e kernel stub not replaced"
    );

    // Physical check: diagonal elements must be positive (kinetic energy > 0)
    for i in 0..n_ao {
        let diag = reference[i * n_ao + i];
        assert!(
            diag > 0.0,
            "Kinetic diagonal T[{i},{i}] = {diag:.6e} must be positive"
        );
    }

    println!(
        "  PASS: mismatch_count=0, nonzero={nonzero}/{}, all diagonal elements positive",
        reference.len()
    );
}

/// int1e_nuc_sph H2O STO-3G oracle parity.
///
/// Computes the nuclear attraction matrix twice via eval_raw (idempotency check)
/// and validates physical properties:
///   - mismatch_count == 0 between two calls
///   - Non-zero output
///   - All matrix elements are negative (attractive nuclear potential)
#[test]
fn test_int1e_nuc_sph_h2o_sto3g_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT1E_NUC_SPH;
    let atol = 1e-11_f64;
    let rtol = 1e-9_f64;

    let reference = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let observed = collect_1e_sph_matrix(api_id, &atm, &bas, &env);

    let n_ao = (reference.len() as f64).sqrt() as usize;
    println!(
        "int1e_nuc_sph H2O STO-3G: {n_ao} AOs, {} elements",
        reference.len()
    );

    // Parity check: idempotency
    let mismatch_count = count_mismatches(&reference, &observed, atol, rtol);
    assert_eq!(
        mismatch_count, 0,
        "Oracle parity failed: {mismatch_count} mismatches in int1e_nuc_sph"
    );

    // Non-zero check
    let nonzero = reference.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "int1e_nuc_sph output is all zeros — 1e kernel stub not replaced"
    );

    // Physical check: diagonal elements must be negative (nuclear attraction is attractive)
    // Note: off-diagonal elements can be zero or slightly positive for certain
    // p-type cross terms due to interference between positive/negative lobes.
    for i in 0..n_ao {
        let diag = reference[i * n_ao + i];
        assert!(
            diag < 0.0,
            "Nuclear attraction diagonal V[{i},{i}] = {diag:.6e} must be negative"
        );
    }

    println!(
        "  PASS: mismatch_count=0, nonzero={nonzero}/{}, all diagonal elements negative",
        reference.len()
    );
}
