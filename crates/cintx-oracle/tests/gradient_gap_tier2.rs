//! Density-fitting Hessian gap oracle gates.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    let mut coord_ptrs = Vec::new();
    for coord in [[-0.4, 0.1, -0.2], [0.5, -0.3, 0.7], [0.2, 0.6, -0.5]] {
        coord_ptrs.push(env.len() as i32);
        env.extend_from_slice(&coord);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.2, 0.8]);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.6, 0.4]);

    let mut atm = vec![0; 3 * ATM_SLOTS];
    for atom in 0..3 {
        let offset = atom * ATM_SLOTS;
        atm[offset + CHARGE_OF] = [6, 8, 7][atom];
        atm[offset + PTR_COORD] = coord_ptrs[atom];
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0; 3 * BAS_SLOTS];
    for (shell, &(atom, l, nctr, coeff)) in [
        (0, 2, 2, d_coeff_ptr),
        (1, 1, 1, p_coeff_ptr),
        (2, 1, 1, p_coeff_ptr),
    ]
    .iter()
    .enumerate()
    {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = atom;
        bas[offset + ANG_OF] = l;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = nctr;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff;
    }
    (atm, bas, env)
}

fn ours(
    api: RawApiId,
    shls: &[i32],
    len: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; len];
    unsafe { eval_raw(api, Some(&mut out), None, shls, atm, bas, env, None, None).unwrap() };
    out
}

#[cfg(has_vendor_libcint)]
fn compare(label: &str, ours: &[f64], vendor: &[f64]) {
    assert!(
        ours.iter().any(|value| value.abs() > 1e-14),
        "{label} is zero"
    );
    let max_abs = ours
        .iter()
        .zip(vendor)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs <= ATOL, "{label}: max_abs={max_abs:.3e}");
}

#[test]
#[cfg(has_vendor_libcint)]
fn vendor_df_hessian_cart_and_sph_d_shell_nctr2() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = fixture();

    for (cart, nd, np) in [(true, 12, 3), (false, 10, 3)] {
        let shls2 = [0, 2];
        let len2 = 9 * nd * np;
        let api2 = if cart {
            RawApiId::INT2C2E_IP1IP2_CART
        } else {
            RawApiId::INT2C2E_IP1IP2_SPH
        };
        let value2 = ours(api2, &shls2, len2, &atm, &bas, &env);
        assert_eq!(value2, ours(api2, &shls2, len2, &atm, &bas, &env));
        let mut vendor2 = vec![0.0; len2];
        if cart {
            vendor_ffi::vendor_int2c2e_ip1ip2_cart(&mut vendor2, &shls2, &atm, 3, &bas, 3, &env);
        } else {
            vendor_ffi::vendor_int2c2e_ip1ip2_sph(&mut vendor2, &shls2, &atm, 3, &bas, 3, &env);
        }
        compare("int2c2e_ip1ip2", &value2, &vendor2);

        let shls3 = [0, 1, 2];
        let len3 = 9 * nd * np * np;
        for (name, api) in [
            (
                "ipvip1",
                if cart {
                    RawApiId::INT3C2E_IPVIP1_CART
                } else {
                    RawApiId::INT3C2E_IPVIP1_SPH
                },
            ),
            (
                "ip1ip2",
                if cart {
                    RawApiId::INT3C2E_IP1IP2_CART
                } else {
                    RawApiId::INT3C2E_IP1IP2_SPH
                },
            ),
        ] {
            let value = ours(api, &shls3, len3, &atm, &bas, &env);
            assert_eq!(value, ours(api, &shls3, len3, &atm, &bas, &env));
            let mut vendor = vec![0.0; len3];
            match (name, cart) {
                ("ipvip1", true) => vendor_ffi::vendor_int3c2e_ipvip1_cart(
                    &mut vendor,
                    &shls3,
                    &atm,
                    3,
                    &bas,
                    3,
                    &env,
                ),
                ("ipvip1", false) => vendor_ffi::vendor_int3c2e_ipvip1_sph(
                    &mut vendor,
                    &shls3,
                    &atm,
                    3,
                    &bas,
                    3,
                    &env,
                ),
                ("ip1ip2", true) => vendor_ffi::vendor_int3c2e_ip1ip2_cart(
                    &mut vendor,
                    &shls3,
                    &atm,
                    3,
                    &bas,
                    3,
                    &env,
                ),
                ("ip1ip2", false) => vendor_ffi::vendor_int3c2e_ip1ip2_sph(
                    &mut vendor,
                    &shls3,
                    &atm,
                    3,
                    &bas,
                    3,
                    &env,
                ),
                _ => unreachable!(),
            };
            compare(&format!("int3c2e_{name}"), &value, &vendor);
        }
    }
}
