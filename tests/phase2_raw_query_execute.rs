use cintx::{
    IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, WorkspaceQueryOptions,
};

#[test]
fn raw_query_null_equivalent_contract() {
    let (atm, bas, env) = sample_raw_layout();
    let operator = one_electron_overlap();
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(1024),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-raw-query"],
    };

    let baseline = cintx::raw::query_workspace_compat(
        operator,
        Representation::Spherical,
        cintx::raw::RawCompatRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect("legacy compat query path should still validate");

    let null_query = cintx::raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Spherical,
        cintx::raw::RawQueryRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: None,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect("null-equivalent query should produce workspace metadata");

    let empty: &[f64] = &[];
    let empty_sentinel_query = cintx::raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Spherical,
        cintx::raw::RawQueryRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: Some(empty),
            cache: Some(empty),
            opt: None,
        },
        &options,
    )
    .expect("empty slices should normalize to null-equivalent query semantics");

    assert_eq!(baseline.required_bytes, null_query.required_bytes);
    assert_eq!(baseline.required_elements, null_query.required_elements);
    assert_eq!(null_query, empty_sentinel_query);
    assert_eq!(null_query.shell_tuple, vec![0, 1]);
    assert_eq!(null_query.dims, vec![1, 3]);
    assert_eq!(null_query.required_elements, 3);
    assert_eq!(null_query.required_bytes, 24);
    assert!(null_query.query_uses_null_out);
    assert!(null_query.query_uses_null_cache);
    assert_eq!(null_query.output_provided_len, 0);
    assert_eq!(null_query.cache_provided_len, 0);
}

#[test]
fn raw_query_then_execute_success() {
    let (atm, bas, env) = sample_raw_layout();
    let operator = one_electron_overlap();
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(1024),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-raw-query", "phase2-raw-execute"],
    };

    let queried = cintx::raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Spherical,
        cintx::raw::RawQueryRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: None,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect("query metadata should succeed for valid raw request");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![0.0f64; required_scalars + 2];
    output[required_scalars] = 1234.0;
    output[required_scalars + 1] = 5678.0;

    let result = cintx::raw::evaluate_compat(
        operator,
        Representation::Spherical,
        &queried,
        cintx::raw::RawEvaluateRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut output,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect("query-compliant raw execute should dispatch successfully");

    assert_eq!(
        result.dispatch.backend,
        cintx::ExecutionBackend::CpuReference
    );
    assert_eq!(result.required_elements, queried.required_elements);
    assert_eq!(result.required_bytes, queried.required_bytes);
    assert_eq!(result.dims, queried.dims);
    assert_eq!(result.cache_used_len, 0);
    assert!(
        output[..required_scalars].iter().any(|value| *value != 0.0),
        "execute path should write deterministic payload into queried output span"
    );
    assert_eq!(
        output[required_scalars], 1234.0,
        "bytes beyond required output span must remain untouched"
    );
    assert_eq!(
        output[required_scalars + 1],
        5678.0,
        "bytes beyond required output span must remain untouched"
    );

    match result.route_target {
        cintx::CpuRouteTarget::Direct(symbol) => {
            assert_eq!(symbol.name(), "int1e_ovlp_sph");
            assert!(!symbol.as_ptr().is_null());
        }
        other => panic!("expected direct CPU route target, got {other:?}"),
    }
}

#[test]
fn raw_query_execute_mismatch_failure_diagnostics() {
    let (atm, bas, env) = sample_raw_layout();
    let operator = one_electron_overlap();
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(1024),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-raw-query", "phase2-raw-execute"],
    };

    let queried = cintx::raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Spherical,
        cintx::raw::RawQueryRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: None,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect("query metadata should succeed for valid request");

    let required_scalars = queried.required_bytes / 8;
    let mut undersized_output = vec![0.0f64; required_scalars - 1];
    let failure = cintx::raw::evaluate_compat(
        operator,
        Representation::Spherical,
        &queried,
        cintx::raw::RawEvaluateRequest {
            shls: &[0, 1],
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut undersized_output,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("execute must fail when output buffer does not satisfy queried requirement");

    assert!(matches!(
        failure.error,
        LibcintRsError::InvalidLayout {
            item: "out_length",
            expected,
            got
        } if expected == required_scalars && got == (required_scalars - 1)
    ));
    assert_eq!(failure.diagnostics.api, "raw.compat.evaluate");
    assert_eq!(failure.diagnostics.shell_tuple, vec![0, 1]);
    assert_eq!(failure.diagnostics.dims, queried.dims);
    assert_eq!(
        failure.diagnostics.required_bytes,
        Some(queried.required_bytes)
    );
    assert_eq!(
        failure.diagnostics.provided_bytes,
        Some((required_scalars - 1) * 8)
    );
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
