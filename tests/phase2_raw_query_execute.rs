use cintx::{IntegralFamily, Operator, OperatorKind, Representation, WorkspaceQueryOptions};

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
