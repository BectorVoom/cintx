//! Spin-dependent two-electron gradient gap oracle gates.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 2e-11;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.push(0.8);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, -0.35]);
    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0; 4 * BAS_SLOTS];
    for shell in 0..4 {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = (shell % 2) as i32;
        bas[offset + ANG_OF] = 1;
        bas[offset + NPRIM_OF] = 1;
        bas[offset + NCTR_OF] = 2;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_two_electron_sigma_gradient_spinor() {
    use cintx_oracle::vendor_ffi as vendor;
    type VendorFn = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;
    let cases: [(&str, VendorFn); 6] = [
        ("ipspsp1", vendor::vendor_int2e_ipspsp1_spinor),
        ("ip1spsp2", vendor::vendor_int2e_ip1spsp2_spinor),
        ("ipspsp1spsp2", vendor::vendor_int2e_ipspsp1spsp2_spinor),
        ("ipsrsr1", vendor::vendor_int2e_ipsrsr1_spinor),
        ("ip1srsr2", vendor::vendor_int2e_ip1srsr2_spinor),
        ("ipsrsr1srsr2", vendor::vendor_int2e_ipsrsr1srsr2_spinor),
    ];
    let (atm, bas, env) = fixture();
    let shls = [0, 1, 2, 3];
    for (op, vendor_fn) in cases {
        let symbol: &'static str = Box::leak(format!("int2e_{op}_spinor").into_boxed_str());
        let len = 3 * 12usize.pow(4) * 2;
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
        let mut reference = vec![0.0; len];
        assert_ne!(vendor_fn(&mut reference, &shls, &atm, 2, &bas, 4, &env), 0);
        let (max_index, max_abs) = ours
            .iter()
            .zip(&reference)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        if max_abs > ATOL {
            let block = 12usize.pow(4) * 2;
            let local = max_index % block;
            eprintln!(
                "{symbol} local={local}: ours_groups={:?} vendor_groups={:?}",
                [ours[local], ours[block + local], ours[2 * block + local]],
                [
                    reference[local],
                    reference[block + local],
                    reference[2 * block + local]
                ]
            );
        }
        assert!(
            max_abs <= ATOL,
            "{symbol}: max_abs={max_abs:.3e} index={max_index} ours={:.6e} vendor={:.6e}",
            ours[max_index],
            reference[max_index],
            // Tensor components occupy contiguous spinor blocks; the primary
            // values above are sufficient for CI while the index identifies
            // any ordering regression.
        );
    }
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_electron_two_sigma_base_spinor() {
    use cintx_oracle::vendor_ffi::vendor_int2e_spsp2_spinor;
    let (atm, bas, env) = fixture();
    let shls = [0, 1, 2, 3];
    let len = 12usize.pow(4) * 2;
    let mut ours = vec![0.0; len];
    unsafe {
        eval_raw(
            RawApiId::Symbol("int2e_spsp2_spinor"),
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
    let mut reference = vec![0.0; len];
    assert_ne!(
        vendor_int2e_spsp2_spinor(&mut reference, &shls, &atm, 2, &bas, 4, &env),
        0
    );
    let max_abs = ours
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs <= ATOL, "int2e_spsp2_spinor: {max_abs:.3e}");
}
