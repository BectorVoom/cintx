//! Oracle parity tests for 4-center 2-electron spherical integrals (`int2e_sph`).
//!
//! Compares cintx `eval_raw(INT2E_SPH)` output against vendored libcint 6.1.3
//! (`vendor_int2e_sph`) across all shell quartets.
//!
//! Tolerance (per Phase 10 RESEARCH D-06):
//! - absolute: 1e-12
//! - relative: 1e-10

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn matches_with_tol(reference: f64, observed: f64, atol: f64, rtol: f64) -> bool {
    let diff = (observed - reference).abs();
    if diff <= atol {
        return true;
    }
    let denom = reference.abs().max(1.0e-15);
    (diff / denom) <= rtol
}

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "output length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (idx, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        if !matches_with_tol(ref_val, obs_val, atol, rtol) {
            mismatches += 1;
            if mismatches <= 16 {
                let diff = (obs_val - ref_val).abs();
                let rel = diff / ref_val.abs().max(1.0e-15);
                eprintln!(
                    "  MISMATCH[{idx}] ref={ref_val:.15e} obs={obs_val:.15e} diff={diff:.3e} rel={rel:.3e}"
                );
            }
        }
    }
    mismatches
}

fn eval_int2e_sph_cintx(
    shls: &[i32; 4],
    n_elem: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_elem];
    unsafe {
        eval_raw(
            RawApiId::INT2E_SPH,
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw int2e_sph failed for shells {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn eval_int2e_sph_vendor(
    shls: &[i32; 4],
    n_elem: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;

    let mut out = vec![0.0_f64; n_elem];
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    vendor_ffi::vendor_int2e_sph(&mut out, shls, atm, natm, bas, nbas, env);
    out
}

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

fn build_h2_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let h1_coord = [0.0_f64, 0.0, -0.7000];
    let h2_coord = [0.0_f64, 0.0, 0.7000];

    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let h1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[0 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[0 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 1;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

#[cfg(has_vendor_libcint)]
fn run_vendor_parity(label: &str, n_shells: usize, atm: &[i32], bas: &[i32], env: &[f64]) {
    let atol = 1.0e-12_f64;
    let rtol = 1.0e-10_f64;

    let ang: Vec<i32> = (0..n_shells).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut quartet_count = 0usize;

    for i_sh in 0..n_shells {
        let ni = shell_nsph[i_sh];
        for j_sh in 0..n_shells {
            let nj = shell_nsph[j_sh];
            for k_sh in 0..n_shells {
                let nk = shell_nsph[k_sh];
                for l_sh in 0..n_shells {
                    quartet_count += 1;
                    let nl = shell_nsph[l_sh];
                    let n_elem = ni * nj * nk * nl;
                    let shls = [i_sh as i32, j_sh as i32, k_sh as i32, l_sh as i32];

                    let vendor_out = eval_int2e_sph_vendor(&shls, n_elem, atm, bas, env);
                    let cintx_out = eval_int2e_sph_cintx(&shls, n_elem, atm, bas, env);

                    mismatch_count += count_mismatches(&vendor_out, &cintx_out, atol, rtol);
                    if cintx_out.iter().any(|v| v.abs() > 1.0e-18) {
                        any_nonzero = true;
                    }
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "{label}: int2e_sph parity mismatches against vendored libcint"
    );
    assert!(
        any_nonzero,
        "{label}: all int2e_sph outputs are zero, kernel appears stubbed"
    );

    println!(
        "{label}: int2e_sph vendor parity PASS over {quartet_count} quartets (atol={atol:.1e}, rtol={rtol:.1e})"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_sph_h2o_sto3g_two_electron() {
    let (atm, bas, env) = build_h2o_sto3g();
    run_vendor_parity("H2O STO-3G", 5, &atm, &bas, &env);
}

#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_sph_h2_sto3g_two_electron() {
    let (atm, bas, env) = build_h2_sto3g();
    run_vendor_parity("H2 STO-3G", 2, &atm, &bas, &env);
}
