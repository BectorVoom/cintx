//! Oracle parity test for 3c2e spherical integrals (int3c2e_sph): H2O STO-3G.
//!
//! Validates the end-to-end compute pipeline for `int3c2e_sph` by comparing:
//! - cintx values via `eval_raw` (dispatches through `launch_center_3c2e`)
//! - reference values from vendored libcint 6.1.3 FFI (when enabled)
//!
//! Tolerance: atol 1e-9 for 3c2e per phase research D-06.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

/// Build H2O STO-3G libcint-style atm/bas/env arrays.
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

    // 2e-family kernels read libcint global env slots (e.g. PTR_RANGE_OMEGA),
    // so user payload must start at PTR_ENV_START to avoid corrupting them.
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32;
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

const N_SHELLS: usize = 5;

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

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
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, diff={diff:.3e}, atol={atol:.1e}"
            );
        }
    }
    mismatches
}

#[test]
fn test_center_3c2e_sph_h2o_sto3g_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C2E_IP1_SPH;

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];

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
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"));
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
                    .unwrap_or_else(|e| panic!("eval_raw second call failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"));
                }

                mismatch_count += count_mismatches(&out1, &out2, 1e-15);
                if out1.iter().any(|&v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "int3c2e_sph idempotency failed: {mismatch_count} mismatches"
    );
    assert!(
        any_nonzero,
        "int3c2e_sph output is all zeros - 3c2e kernel stub not replaced"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_center_3c2e_sph_h2o_sto3g_vendor_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C2E_IP1_SPH;
    let atol = 1e-9_f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

                let mut vendor_out = vec![0.0_f64; n_elem];
                let mut cintx_out = vec![0.0_f64; n_elem];

                vendor_ffi::vendor_int3c2e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

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
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"));
                }

                if vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    || cintx_out.iter().any(|&v| v.abs() > 1e-18)
                {
                    any_nonzero = true;
                }

                mismatch_count += count_mismatches(&vendor_out, &cintx_out, atol);
            }
        }
    }

    assert!(any_nonzero, "int3c2e_sph outputs are all zeros");
    assert_eq!(
        mismatch_count, 0,
        "int3c2e_sph vendor parity failed: {mismatch_count} elements exceed atol=1e-9"
    );
}
