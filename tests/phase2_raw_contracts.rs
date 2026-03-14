use cintx::runtime::raw::{RawAtmView, RawBasView, RawEnvView};
use cintx::LibcintRsError;

#[test]
fn raw_layout_slot_and_offset_checks() {
    let (atm, bas, env) = sample_raw_layout();
    let env_view = RawEnvView::new(&env);

    let invalid_atm = RawAtmView::new(&atm[..7]).expect_err("atm length must align to ATM_SLOTS");
    assert!(matches!(
        invalid_atm,
        LibcintRsError::InvalidInput { field: "atm", .. }
    ));

    let invalid_bas = RawBasView::new(&bas[..9]).expect_err("bas length must align to BAS_SLOTS");
    assert!(matches!(
        invalid_bas,
        LibcintRsError::InvalidInput { field: "bas", .. }
    ));

    let mut bad_atm = atm.clone();
    bad_atm[1] = env.len() as i32 - 1;
    let atm_view = RawAtmView::new(&bad_atm).expect("atm divisibility should still hold");
    let bad_coord = atm_view
        .validate_offsets(&env_view)
        .expect_err("coordinate pointer must be range-checked");
    assert!(matches!(
        bad_coord,
        LibcintRsError::InvalidInput {
            field: "atm.ptr_coord",
            ..
        }
    ));

    let mut bad_bas = bas.clone();
    bad_bas[5] = -1;
    let bas_view = RawBasView::new(&bad_bas).expect("bas divisibility should still hold");
    let bad_exp = bas_view
        .validate_offsets(2, &env_view)
        .expect_err("negative exponent offset must fail");
    assert!(matches!(
        bad_exp,
        LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            ..
        }
    ));
}

fn sample_raw_layout() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let atm = vec![
        8, 20, 1, 0, 0, 0, //
        1, 23, 1, 0, 0, 0,
    ];
    let bas = vec![
        0, 0, 2, 1, 0, 28, 30, 0, //
        1, 1, 2, 1, 0, 32, 34, 0,
    ];
    let mut env = vec![0.0f64; 40];
    env[20..23].copy_from_slice(&[0.0, 0.0, -0.1173]);
    env[23..26].copy_from_slice(&[0.0, 0.7572, 0.4692]);
    env[28..30].copy_from_slice(&[130.70932, 5.0331513]);
    env[30..32].copy_from_slice(&[0.154329, 0.535328]);
    env[32..34].copy_from_slice(&[3.42525091, 0.62391373]);
    env[34..36].copy_from_slice(&[0.154329, 0.535328]);

    (atm, bas, env)
}
