use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, Shell,
    WorkspaceQueryOptions,
};

#[test]
fn allocation_failures_are_typed() {
    let safe_basis = sample_safe_basis();
    let operator = one_electron_overlap();
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(1024),
        backend_candidate: "cpu",
        feature_flags: vec!["simulate-allocation-failure", "phase2-allocation-failures"],
    };

    let safe_failure = cintx::safe::evaluate(
        &safe_basis,
        operator,
        Representation::Spherical,
        &[0, 1],
        &options,
    )
    .expect_err("safe evaluate should surface simulated allocation failure as typed error");
    assert!(matches!(
        safe_failure.error,
        LibcintRsError::AllocationFailure {
            operation: "safe.evaluate",
            ..
        }
    ));
    assert_eq!(safe_failure.diagnostics.api, "safe.evaluate");
    assert_eq!(safe_failure.diagnostics.shell_tuple, vec![0, 1]);
    assert_eq!(safe_failure.diagnostics.memory_limit_bytes, Some(1024));
    assert!(
        safe_failure
            .diagnostics
            .feature_flags
            .contains(&"simulate-allocation-failure")
    );
    assert_ne!(safe_failure.diagnostics.correlation_id, 0);

    let (atm, bas, env) = sample_raw_layout();
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
    .expect("raw query should succeed before simulated allocation failure at execute");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![0.0f64; required_scalars];
    let raw_failure = cintx::raw::evaluate_compat(
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
    .expect_err("raw evaluate should surface simulated allocation failure as typed error");
    assert!(matches!(
        raw_failure.error,
        LibcintRsError::AllocationFailure {
            operation: "raw.compat.evaluate",
            ..
        }
    ));
    assert_eq!(raw_failure.diagnostics.api, "raw.compat.evaluate");
    assert_eq!(raw_failure.diagnostics.shell_tuple, vec![0, 1]);
    assert_eq!(
        raw_failure.diagnostics.required_bytes,
        Some(queried.memory_required_bytes)
    );
    assert_eq!(
        raw_failure.diagnostics.provided_bytes,
        Some(required_scalars * 8)
    );
    assert_eq!(raw_failure.diagnostics.memory_limit_bytes, Some(1024));
    assert!(
        raw_failure
            .diagnostics
            .feature_flags
            .contains(&"simulate-allocation-failure")
    );
    assert_ne!(raw_failure.diagnostics.correlation_id, 0);
}

fn sample_safe_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");
    let shell_d =
        Shell::new(0, 2, vec![4.0, 1.0], vec![0.7, 0.3]).expect("d shell should be valid");
    let shell_p =
        Shell::new(1, 1, vec![3.0, 0.8], vec![0.6, 0.4]).expect("p shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_d, shell_p]).expect("basis should be valid")
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
