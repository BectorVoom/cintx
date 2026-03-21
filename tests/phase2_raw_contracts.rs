use std::ptr::NonNull;

use cintx::runtime::raw::{
    RawAtmView, RawBasView, RawEnvView, RawValidationRequest, validate_raw_contract,
};
use cintx::{
    IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, WorkspaceQueryOptions,
};

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

#[test]
fn raw_validation_matrix() {
    let (atm, bas, env) = sample_raw_layout();
    let operator = one_electron_overlap();

    let valid = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: None,
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: None,
        opt: None,
    })
    .expect("valid raw contract should pass");
    assert_eq!(valid.shell_tuple, vec![0, 1]);
    assert_eq!(valid.natural_dims, vec![1, 3]);
    assert_eq!(valid.dims, vec![1, 3]);
    assert_eq!(valid.required_elements, 3);

    let arity_error = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0],
        dims: None,
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: None,
        opt: None,
    })
    .expect_err("shls arity mismatch should fail");
    assert!(matches!(
        arity_error,
        LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: 1
        }
    ));

    let dims_arity_error = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: Some(&[1]),
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: None,
        opt: None,
    })
    .expect_err("dims arity mismatch should fail");
    assert!(matches!(
        dims_arity_error,
        LibcintRsError::InvalidLayout {
            item: "dims_arity",
            expected: 2,
            got: 1
        }
    ));

    let dims_value_error = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: Some(&[9, 3]),
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: None,
        opt: None,
    })
    .expect_err("dims value mismatch should fail");
    assert!(matches!(
        dims_value_error,
        LibcintRsError::DimsBufferMismatch {
            expected,
            provided
        } if expected == vec![1, 3] && provided == vec![9, 3]
    ));

    let opt_without_cache = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: None,
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: None,
        opt: Some(NonNull::dangling()),
    })
    .expect_err("opt requires cache");
    assert!(matches!(
        opt_without_cache,
        LibcintRsError::InvalidInput { field: "cache", .. }
    ));

    let short_cache = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: None,
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: Some(&[0.0]),
        opt: Some(NonNull::dangling()),
    })
    .expect_err("cache must satisfy validator minimum length");
    assert!(matches!(
        short_cache,
        LibcintRsError::InvalidLayout {
            item: "cache_length",
            expected: 2,
            got: 1
        }
    ));

    let valid_with_cache = validate_raw_contract(RawValidationRequest {
        operator,
        representation: Representation::Spherical,
        shls: &[0, 1],
        dims: None,
        atm: &atm,
        bas: &bas,
        env: &env,
        cache: Some(&[0.0, 0.0]),
        opt: Some(NonNull::dangling()),
    })
    .expect("opt+cache should satisfy validation contract");
    assert!(valid_with_cache.has_cache);
    assert!(valid_with_cache.has_opt);
    assert_eq!(valid_with_cache.cache_required_len, 2);
}

#[test]
fn raw_api_validation_boundary() {
    let (atm, bas, env) = sample_raw_layout();
    let operator = one_electron_overlap();
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(1024),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-raw-contract"],
    };

    let workspace = cintx::raw::query_workspace_compat(
        operator,
        Representation::Spherical,
        cintx::raw::RawCompatRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            cache: Some(&[0.0, 0.0]),
            opt: Some(NonNull::dangling()),
        },
        &options,
    )
    .expect("valid raw compatibility request should pass API boundary");
    assert_eq!(workspace.shell_tuple, vec![0, 1]);
    assert_eq!(workspace.natural_dims, vec![1, 3]);
    assert_eq!(workspace.dims, vec![1, 3]);
    assert_eq!(workspace.required_elements, 3);
    assert_eq!(workspace.required_bytes, 24);
    assert_eq!(
        workspace.cache_required_len, 0,
        "raw.compat query must expose route-backed cache sizing (1e routes are not optimizer/cached)",
    );
    assert!(workspace.has_cache);
    assert!(workspace.has_opt);

    let failure = cintx::raw::query_workspace_compat(
        operator,
        Representation::Spherical,
        cintx::raw::RawCompatRequest {
            shls: &[0, 1],
            dims: Some(&[999, 3]),
            atm: &atm,
            bas: &bas,
            env: &env,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("mismatched dims should fail before execution dispatch");
    assert!(matches!(
        failure.error,
        LibcintRsError::DimsBufferMismatch { .. }
    ));
    assert_eq!(failure.diagnostics.api, "raw.compat.query_workspace");
    assert_eq!(failure.diagnostics.dims, vec![999, 3]);
    assert!(failure.diagnostics.provided_bytes.is_some());
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

fn one_electron_overlap() -> Operator {
    Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap operator should be valid")
}
