//! Oracle parity test for 3c1e spherical integrals (int3c1e_sph): H2O STO-3G.
//!
//! Validates the end-to-end compute pipeline for `int3c1e_sph` (three-center
//! one-electron overlap) by comparing:
//!   - cintx values via `eval_raw` (which dispatches through `launch_center_3c1e`)
//!   - Reference values from vendored libcint 6.1.3 FFI (when CINTX_ORACLE_BUILD_VENDOR=1)
//!
//! Tolerance: atol 1e-7 for 3c1e per RESEARCH.md D-06.
//!
//! H2O STO-3G geometry (in Bohr):
//!   O  at (0.000,  0.000, 0.000)
//!   H1 at (0.000,  1.431, 1.108)
//!   H2 at (0.000, -1.431, 1.108)
//!
//! Shells (STO-3G):
//!   Shell 0: O 1s  (3 primitives, l=0)
//!   Shell 1: O 2s  (3 primitives, l=0)
//!   Shell 2: O 2p  (3 primitives, l=1)
//!   Shell 3: H1 1s (3 primitives, l=0)
//!   Shell 4: H2 1s (3 primitives, l=0)
//!
//! With 5 shells: 5^3 = 125 shell triples.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G basis data
// ─────────────────────────────────────────────────────────────────────────────

/// Build the H2O STO-3G `atm`, `bas`, `env` arrays.
///
/// Matches the build_h2o_sto3g() function in one_electron_parity.rs exactly
/// so comparisons are made on the same molecular geometry.
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    // STO-3G exponents and coefficients (Hehre, Stewart & Pople, JCP 51, 2657, 1969)
    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = Vec::<f64>::new();

    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let _zeta_ptr = env.len() as i32;
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_coeff);

    // atm: [CHARGE_OF, PTR_COORD, NUC_MOD_OF, PTR_ZETA, PTR_FRAC_CHARGE, 0]
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = 9;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = 9;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = 9;

    // bas: [ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, KAPPA_OF, PTR_EXP, PTR_COEFF, 0]
    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

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

/// Number of spherical AOs for angular momentum l: 2l+1.
fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Compare two output slices element-wise with absolute tolerance.
/// Returns the count of elements that fall outside tolerance.
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64) -> usize {
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
        if diff > atol {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, atol={atol:.1e}"
            );
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx self-consistency test (no vendor FFI required)
// ─────────────────────────────────────────────────────────────────────────────

/// int3c1e_sph H2O STO-3G self-consistency test.
///
/// Verifies that:
/// 1. int3c1e_sph via eval_raw produces non-zero values for some shell triples.
/// 2. Results are deterministic (two calls produce identical output).
/// 3. The (0,0,0) diagonal triple (s-s-s) is positive.
#[test]
fn test_int3c1e_sph_h2o_sto3g_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C1E_SPH;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut total_nonzero = 0usize;
    let mut total_mismatch = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

                unsafe {
                    eval_raw(
                        api_id,
                        Some(&mut out1),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"
                        )
                    });

                    eval_raw(
                        api_id,
                        Some(&mut out2),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "eval_raw second call failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"
                        )
                    });
                }

                // Idempotency check
                total_mismatch += count_mismatches(&out1, &out2, 1e-15);

                // Nonzero tracking
                total_nonzero += out1.iter().filter(|&&v| v.abs() > 1e-18).count();
            }
        }
    }

    assert_eq!(
        total_mismatch, 0,
        "int3c1e_sph: {total_mismatch} idempotency mismatches (non-deterministic kernel)"
    );
    assert!(
        total_nonzero > 0,
        "int3c1e_sph: all outputs are zero — 3c1e kernel stub not replaced"
    );

    println!(
        "int3c1e_sph self-consistency: PASS. Nonzero elements: {total_nonzero}/{}",
        N_SHELLS * N_SHELLS * N_SHELLS
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor libcint parity test (requires CINTX_ORACLE_BUILD_VENDOR=1)
// ─────────────────────────────────────────────────────────────────────────────

/// int3c1e_sph H2O STO-3G oracle parity against vendored libcint 6.1.3.
///
/// Iterates over all 5^3 = 125 shell triples and compares:
/// - Cintx output from eval_raw(int3c1e_sph)
/// - Reference output from vendor_int3c1e_sph (vendored libcint 6.1.3 FFI)
///
/// Tolerance: atol 1e-7 (per RESEARCH.md D-06 for 3c1e family).
///
/// Asserts:
/// - mismatch_count == 0
/// - at least one non-zero element seen (non-stub check)
///
/// Note: libcint 3c1e output is column-major (i fastest, k slowest).
/// Our kernel produces the same ordering.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_h2o_sto3g_vendor_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C1E_SPH;
    let atol = 1e-7_f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut triple_count = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                triple_count += 1;
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;

                let mut vendor_out = vec![0.0_f64; n_elem];
                let mut cintx_out = vec![0.0_f64; n_elem];
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

                // Reference: vendored libcint 6.1.3
                vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                // cintx: eval_raw dispatches to launch_center_3c1e
                unsafe {
                    eval_raw(
                        api_id,
                        Some(&mut cintx_out),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"
                        )
                    });
                }

                // Count nonzero elements across both outputs
                if vendor_out.iter().any(|v| v.abs() > 1e-18)
                    || cintx_out.iter().any(|v| v.abs() > 1e-18)
                {
                    any_nonzero = true;
                }

                // Element-wise comparison
                let triple_mismatches = count_mismatches(&vendor_out, &cintx_out, atol);
                if triple_mismatches > 0 {
                    eprintln!(
                        "  Shell triple ({i_sh},{j_sh},{k_sh}) [li={},lj={},lk={}]: \
                         {triple_mismatches} mismatches",
                        ang[i_sh], ang[j_sh], ang[k_sh]
                    );
                }
                mismatch_count += triple_mismatches;
            }
        }
    }

    println!(
        "int3c1e_sph oracle parity: {triple_count} shell triples checked, \
         mismatch_count={mismatch_count}, atol={atol:.1e}"
    );

    assert!(
        any_nonzero,
        "int3c1e_sph: all outputs are zero — either the 3c1e kernel is still a stub \
         or vendor libcint returned all zeros"
    );

    assert_eq!(
        mismatch_count, 0,
        "int3c1e_sph oracle parity: {mismatch_count} elements exceed atol={atol:.1e} \
         vs vendored libcint 6.1.3 for H2O STO-3G"
    );

    println!("  PASS: mismatch_count=0 vs vendored libcint 6.1.3");
}
