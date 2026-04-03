//! Oracle parity tests for 2c2e spherical integrals: H2O STO-3G.
//!
//! Validates the `int2c2e_sph` compute pipeline by comparing:
//!   - Idempotency check: two calls to eval_raw must agree exactly
//!   - [With vendor build] Oracle comparison against vendored libcint 6.1.3
//!
//! H2O STO-3G geometry (in Bohr):
//!   O  at (0.000,  0.000, 0.000)
//!   H1 at (0.000,  1.4307, 1.1078)
//!   H2 at (0.000, -1.4307, 1.1078)
//!
//! Shells (STO-3G):
//!   Shell 0: O 1s  (3 primitives, l=0)
//!   Shell 1: O 2s  (3 primitives, l=0)
//!   Shell 2: O 2p  (3 primitives, l=1)
//!   Shell 3: H1 1s (3 primitives, l=0)
//!   Shell 4: H2 1s (3 primitives, l=0)
//!
//! These tests require the `cpu` feature to be enabled (cubecl cpu backend).
//! Vendor parity tests require CINTX_ORACLE_BUILD_VENDOR=1.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP, PTR_ENV_START, PTR_ZETA, NUC_MOD_OF, POINT_NUC, RawApiId, eval_raw,
};

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G basis data (shared with one_electron_parity.rs)
// ─────────────────────────────────────────────────────────────────────────────

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

    // ── Build env array ──────────────────────────────────────────────────────
    // CRITICAL: libcint reserves env[0..PTR_ENV_START] for global parameters:
    //   PTR_EXPCUTOFF=0, PTR_COMMON_ORIG=1..3, PTR_RINV_ORIG=4..6,
    //   PTR_RINV_ZETA=7, PTR_RANGE_OMEGA=8, PTR_F12_ZETA=9, PTR_GTG_ZETA=10,
    //   PTR_GRIDS=12..19.
    // User data MUST start at PTR_ENV_START=20. Placing coordinates in
    // env[0..19] corrupts e.g. PTR_RANGE_OMEGA, causing range-separated
    // Coulomb kernels to apply omega != 0 and produce wrong 2e+ integrals.
    //
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

const N_SHELLS: usize = 5;

/// Number of spherical AOs for angular momentum l: 2l+1.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Compare two slices element-wise with absolute tolerance.
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
                 diff={diff:.3e}, atol={atol:.3e}"
            );
        }
    }
    mismatches
}

/// Evaluate int2c2e_sph for a shell pair (i_sh, k_sh) using cintx eval_raw.
///
/// Returns the flat output buffer of size ni*nk (i fastest, k slowest).
fn eval_2c2e_sph_cintx(
    i_sh: usize,
    k_sh: usize,
    ni: usize,
    nk: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let shls = [i_sh as i32, k_sh as i32];
    let mut out = vec![0.0_f64; ni * nk];
    unsafe {
        eval_raw(
            RawApiId::INT2C2E_SPH,
            Some(&mut out),
            None,
            &shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw int2c2e_sph failed for shells ({i_sh},{k_sh}): {e:?}"));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Idempotency parity test
// ─────────────────────────────────────────────────────────────────────────────

/// int2c2e_sph H2O STO-3G idempotency parity.
///
/// Calls eval_raw twice for all shell pairs and verifies identical results.
/// Also checks that the output contains non-zero values (real kernel, not stub).
#[test]
fn test_int2c2e_sph_h2o_sto3g_idempotency() {
    let (atm, bas, env) = build_h2o_sto3g();

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let atol = 1e-15_f64;
    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for i_sh in 0..N_SHELLS {
        let ni = shell_nao[i_sh];
        for k_sh in 0..N_SHELLS {
            let nk = shell_nao[k_sh];

            let ref_out = eval_2c2e_sph_cintx(i_sh, k_sh, ni, nk, &atm, &bas, &env);
            let obs_out = eval_2c2e_sph_cintx(i_sh, k_sh, ni, nk, &atm, &bas, &env);

            mismatch_count += count_mismatches(&ref_out, &obs_out, atol);
            if ref_out.iter().any(|&v| v.abs() > 1e-18) {
                any_nonzero = true;
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "int2c2e_sph idempotency failed: {mismatch_count} mismatches"
    );
    assert!(
        any_nonzero,
        "int2c2e_sph output is all zeros — 2c2e kernel stub not replaced"
    );

    println!("  PASS: int2c2e_sph H2O STO-3G idempotency: mismatch_count=0, non-zero values present");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor libcint 6.1.3 oracle parity test
// ─────────────────────────────────────────────────────────────────────────────

/// int2c2e_sph H2O STO-3G vendor libcint oracle parity.
///
/// Compares cintx eval_raw output against vendored libcint 6.1.3 for all
/// shell pairs in H2O STO-3G. Tolerance: atol 1e-9 per D-06.
///
/// libcint 2c2e output is column-major (i fastest, k slowest),
/// matching our eval_raw layout — no transposition needed.
///
/// This test requires CINTX_ORACLE_BUILD_VENDOR=1.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int2c2e_sph_h2o_sto3g_vendor_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS)
        .map(|s| bas[s * BAS_SLOTS + ANG_OF])
        .collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    let atol = 1e-9_f64;
    let mut mismatch_count = 0usize;
    let mut any_nonzero_vendor = false;
    let mut any_nonzero_cintx = false;

    for i_sh in 0..N_SHELLS {
        let ni = shell_nao[i_sh];
        for k_sh in 0..N_SHELLS {
            let nk = shell_nao[k_sh];
            let n_elem = ni * nk;

            // Vendored libcint reference
            let mut vendor_out = vec![0.0_f64; n_elem];
            let shls_arr = [i_sh as i32, k_sh as i32];
            let _ret = vendor_ffi::vendor_int2c2e_sph(
                &mut vendor_out,
                &shls_arr,
                &atm,
                natm,
                &bas,
                nbas,
                &env,
            );

            // cintx eval_raw
            let cintx_out = eval_2c2e_sph_cintx(i_sh, k_sh, ni, nk, &atm, &bas, &env);

            // libcint 2c2e output is column-major (i fastest = same as cintx).
            // Compare directly without transposition.
            // Note: libcint output[j*ni + i] vs cintx output[i + j*ni] — identical layout.
            let mismatches_this = count_mismatches(&vendor_out, &cintx_out, atol);
            if mismatches_this > 0 {
                eprintln!(
                    "  Shell pair ({i_sh}, {k_sh}) ang=({}, {}): {mismatches_this} mismatches",
                    ang[i_sh], ang[k_sh]
                );
                // Print first few values for diagnosis
                for idx in 0..n_elem.min(9) {
                    let diff = (cintx_out[idx] - vendor_out[idx]).abs();
                    if diff > 1e-12 {
                        eprintln!(
                            "    [{}]: vendor={:.15e} cintx={:.15e} diff={:.3e}",
                            idx, vendor_out[idx], cintx_out[idx], diff
                        );
                    }
                }
            }
            mismatch_count += mismatches_this;

            if vendor_out.iter().any(|&v| v.abs() > 1e-18) {
                any_nonzero_vendor = true;
            }
            if cintx_out.iter().any(|&v| v.abs() > 1e-18) {
                any_nonzero_cintx = true;
            }
        }
    }

    assert!(
        any_nonzero_vendor,
        "Vendor int2c2e_sph output is all zeros — vendor build issue"
    );
    assert!(
        any_nonzero_cintx,
        "Cintx int2c2e_sph output is all zeros — 2c2e kernel stub not replaced"
    );

    assert_eq!(
        mismatch_count, 0,
        "int2c2e_sph vendor parity failed: {mismatch_count} elements exceed atol={atol:.0e} \
        vs vendored libcint 6.1.3 for H2O STO-3G"
    );

    println!(
        "  PASS: int2c2e_sph H2O STO-3G vendor parity: mismatch_count=0 at atol=1e-9 \
        vs vendored libcint 6.1.3"
    );
}
