//! Tier-1 four-center gradient-gap oracle gate.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const NCOMP: usize = 81;
const ATOL: f64 = 1e-12;

fn d_contracted_quartet() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let d_exp = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let d_coeff = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.2, 0.8]);
    let s_exp = env.len() as i32;
    env.extend_from_slice(&[1.3, 0.38]);
    let s_coeff = env.len() as i32;
    env.extend_from_slice(&[0.6, 0.4]);

    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0; 4 * BAS_SLOTS];
    let specs = [
        (0, 2, 2, 2, d_exp, d_coeff),
        (1, 0, 2, 1, s_exp, s_coeff),
        (0, 0, 2, 1, s_exp, s_coeff),
        (1, 0, 2, 1, s_exp, s_coeff),
    ];
    for (shell, &(atom, l, nprim, nctr, exp, coeff)) in specs.iter().enumerate() {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = atom;
        bas[offset + ANG_OF] = l;
        bas[offset + NPRIM_OF] = nprim;
        bas[offset + NCTR_OF] = nctr;
        bas[offset + PTR_EXP] = exp;
        bas[offset + PTR_COEFF] = coeff;
    }
    (atm, bas, env)
}

#[cfg(has_vendor_libcint)]
fn assert_parity(cart: bool) {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = d_contracted_quartet();
    let shls = [0, 1, 2, 3];
    let nao_d = if cart { 12 } else { 10 };
    let len = NCOMP * nao_d;
    let api = if cart {
        RawApiId::INT2E_IPVIP1IPVIP2_CART
    } else {
        RawApiId::INT2E_IPVIP1IPVIP2_SPH
    };
    let mut ours = vec![0.0; len];
    unsafe {
        eval_raw(
            api,
            Some(&mut ours),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .expect("cintx int2e_ipvip1ipvip2 evaluation");
    }
    let mut repeated = vec![0.0; len];
    unsafe {
        eval_raw(
            api,
            Some(&mut repeated),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .expect("repeated cintx int2e_ipvip1ipvip2 evaluation");
    }
    assert_eq!(
        ours, repeated,
        "four-center derivative must be deterministic"
    );
    assert!(ours.iter().any(|value| value.abs() > 1e-14));

    let mut vendor = vec![0.0; len];
    let not0 = if cart {
        vendor_ffi::vendor_int2e_ipvip1ipvip2_cart(&mut vendor, &shls, &atm, 2, &bas, 4, &env)
    } else {
        vendor_ffi::vendor_int2e_ipvip1ipvip2_sph(&mut vendor, &shls, &atm, 2, &bas, 4, &env)
    };
    assert_ne!(not0, 0);
    let max_abs = ours
        .iter()
        .zip(&vendor)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs <= ATOL, "max_abs={max_abs:.3e} > {ATOL:.1e}");
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int2e_ipvip1ipvip2_cart_d_shell_nctr2() {
    assert_parity(true);
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int2e_ipvip1ipvip2_sph_d_shell_nctr2() {
    assert_parity(false);
}
