//! Scalar X2C gradient/Hessian gap oracle gates.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    env[PTR_RINV_ORIG..PTR_RINV_ORIG + 3].copy_from_slice(&[0.31, -0.27, 0.19]);
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let a_coeff = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.2, 0.8]);
    let b_coeff = env.len() as i32;
    env.extend_from_slice(&[0.6, 0.4, 0.25, -0.75]);
    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0; 2 * BAS_SLOTS];
    for (offset, atom, coeff) in [(0, 0, a_coeff), (BAS_SLOTS, 1, b_coeff)] {
        bas[offset + ATOM_OF] = atom;
        bas[offset + ANG_OF] = 2;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = 2;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff;
    }
    (atm, bas, env)
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_scalar_x2c_gap_all_representations() {
    use cintx_oracle::vendor_ffi as v;
    type VFn = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;
    let cases: [(&str, usize, VFn, VFn, VFn); 6] = [
        (
            "ippnucp",
            3,
            v::vendor_int1e_ippnucp_cart,
            v::vendor_int1e_ippnucp_sph,
            v::vendor_int1e_ippnucp_spinor,
        ),
        (
            "ipprinvp",
            3,
            v::vendor_int1e_ipprinvp_cart,
            v::vendor_int1e_ipprinvp_sph,
            v::vendor_int1e_ipprinvp_spinor,
        ),
        (
            "ippnucpip",
            9,
            v::vendor_int1e_ippnucpip_cart,
            v::vendor_int1e_ippnucpip_sph,
            v::vendor_int1e_ippnucpip_spinor,
        ),
        (
            "ipprinvpip",
            9,
            v::vendor_int1e_ipprinvpip_cart,
            v::vendor_int1e_ipprinvpip_sph,
            v::vendor_int1e_ipprinvpip_spinor,
        ),
        (
            "ipippnucp",
            9,
            v::vendor_int1e_ipippnucp_cart,
            v::vendor_int1e_ipippnucp_sph,
            v::vendor_int1e_ipippnucp_spinor,
        ),
        (
            "ipipprinvp",
            9,
            v::vendor_int1e_ipipprinvp_cart,
            v::vendor_int1e_ipipprinvp_sph,
            v::vendor_int1e_ipipprinvp_spinor,
        ),
    ];
    let (atm, bas, env) = fixture();
    let shls = [0, 1];
    for (op, rank, cart_fn, sph_fn, spinor_fn) in cases {
        for (rep, nao, scalar_lanes, vendor_fn) in [
            ("cart", 12, 1, cart_fn),
            ("sph", 10, 1, sph_fn),
            ("spinor", 20, 2, spinor_fn),
        ] {
            let symbol: &'static str = Box::leak(format!("int1e_{op}_{rep}").into_boxed_str());
            let len = rank * nao * nao * scalar_lanes;
            let mut ours = vec![0.0; len];
            unsafe {
                eval_raw(
                    RawApiId::Symbol(symbol),
                    Some(&mut ours),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }
            let mut repeated = vec![0.0; len];
            unsafe {
                eval_raw(
                    RawApiId::Symbol(symbol),
                    Some(&mut repeated),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }
            assert_eq!(ours, repeated, "{symbol} determinism");
            let mut vendor = vec![0.0; len];
            assert_ne!(vendor_fn(&mut vendor, &shls, &atm, 2, &bas, 2, &env), 0);
            let max_abs = ours
                .iter()
                .zip(&vendor)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_abs <= ATOL, "{symbol}: max_abs={max_abs:.3e}");
        }
    }
}
