//! Tier-1 gradient-gap oracle gates.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA,
    RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn d_shell_nctr2_with_rinv_origin() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    env[PTR_RINV_ORIG] = 0.31;
    env[PTR_RINV_ORIG + 1] = -0.27;
    env[PTR_RINV_ORIG + 2] = 0.19;
    env[PTR_COMMON_ORIG] = -0.23;
    env[PTR_COMMON_ORIG + 1] = 0.17;
    env[PTR_COMMON_ORIG + 2] = 0.29;

    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let a_exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let a_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.2, 0.8]);
    let b_exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.3, 0.38]);
    let b_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.6, 0.4, 0.25, -0.75]);

    let mut atm = vec![0; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 6;
    atm[PTR_COORD] = a_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 8;
    atm[ATM_SLOTS + PTR_COORD] = b_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0; 2 * BAS_SLOTS];
    for (offset, atom, exp_ptr, coeff_ptr) in [
        (0, 0, a_exp_ptr, a_coeff_ptr),
        (BAS_SLOTS, 1, b_exp_ptr, b_coeff_ptr),
    ] {
        bas[offset + ATOM_OF] = atom;
        bas[offset + ANG_OF] = 2;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = 2;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn eval_cintx(symbol: &'static str, len: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; len];
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            &[0, 1],
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|error| panic!("{symbol} failed: {error:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn assert_parity(operator: &str, representation: &str, nao: usize) {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = d_shell_nctr2_with_rinv_origin();
    let symbol = match (operator, representation) {
        ("iprinvip", "cart") => "int1e_iprinvip_cart",
        ("iprinvip", "sph") => "int1e_iprinvip_sph",
        ("iprinvip", "spinor") => "int1e_iprinvip_spinor",
        ("ipipr", "cart") => "int1e_ipipr_cart",
        ("ipipr", "sph") => "int1e_ipipr_sph",
        ("ipipr", "spinor") => "int1e_ipipr_spinor",
        _ => unreachable!(),
    };
    let scalar_lanes = if representation == "spinor" { 2 } else { 1 };
    let rank = if operator == "ipipr" { 27 } else { 9 };
    let len = rank * nao * nao * scalar_lanes;
    let ours = eval_cintx(symbol, len, &atm, &bas, &env);
    let repeated = eval_cintx(symbol, len, &atm, &bas, &env);
    assert_eq!(ours, repeated, "{symbol} must be deterministic");

    let mut vendor = vec![0.0; len];
    let shls = [0, 1];
    let not0 = match (operator, representation) {
        ("iprinvip", "cart") => {
            vendor_ffi::vendor_int1e_iprinvip_cart(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        ("iprinvip", "sph") => {
            vendor_ffi::vendor_int1e_iprinvip_sph(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        ("iprinvip", "spinor") => {
            vendor_ffi::vendor_int1e_iprinvip_spinor(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        ("ipipr", "cart") => {
            vendor_ffi::vendor_int1e_ipipr_cart(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        ("ipipr", "sph") => {
            vendor_ffi::vendor_int1e_ipipr_sph(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        ("ipipr", "spinor") => {
            vendor_ffi::vendor_int1e_ipipr_spinor(&mut vendor, &shls, &atm, 2, &bas, 2, &env)
        }
        _ => unreachable!(),
    };
    assert_ne!(not0, 0, "vendor {symbol} unexpectedly screened to zero");
    let max_abs = ours
        .iter()
        .zip(&vendor)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= ATOL,
        "{symbol} max_abs={max_abs:.3e} > {ATOL:.1e}"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_iprinvip_cart_d_shell_nctr2() {
    assert_parity("iprinvip", "cart", 12);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_iprinvip_sph_d_shell_nctr2() {
    assert_parity("iprinvip", "sph", 10);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_iprinvip_spinor_d_shell_nctr2() {
    assert_parity("iprinvip", "spinor", 20);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_ipipr_cart_d_shell_nctr2() {
    assert_parity("ipipr", "cart", 12);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_ipipr_sph_d_shell_nctr2() {
    assert_parity("ipipr", "sph", 10);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_ipipr_spinor_d_shell_nctr2() {
    assert_parity("ipipr", "spinor", 20);
}
