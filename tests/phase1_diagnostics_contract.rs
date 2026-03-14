use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, Shell,
    WorkspaceQueryOptions, query_workspace_safe,
};

#[test]
fn diagnostics_fields_complete() {
    let basis = sample_basis();
    let operator =
        Operator::new(IntegralFamily::OneElectron, OperatorKind::Kinetic).expect("valid operator");
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(64),
        backend_candidate: "cpu",
        feature_flags: vec!["with-f12"],
    };

    let failure = query_workspace_safe(
        &basis,
        operator,
        Representation::Spherical,
        &[0, 1],
        &options,
    )
    .expect_err("memory cap should force typed failure");

    assert!(matches!(
        failure.error,
        LibcintRsError::MemoryLimitExceeded { .. }
    ));
    assert_eq!(failure.diagnostics.api, "safe.query_workspace");
    assert_eq!(
        failure.diagnostics.representation,
        Representation::Spherical.as_str()
    );
    assert_eq!(failure.diagnostics.shell_tuple, vec![0, 1]);
    assert!(!failure.diagnostics.dims.is_empty());
    assert!(failure.diagnostics.required_bytes.is_some());
    assert!(failure.diagnostics.provided_bytes.is_some());
    assert_eq!(failure.diagnostics.memory_limit_bytes, Some(64));
    assert_eq!(failure.diagnostics.backend_candidate, "cpu");
    assert_eq!(failure.diagnostics.feature_flags, vec!["with-f12"]);
    assert_ne!(failure.diagnostics.correlation_id, 0);
}

#[test]
fn validation_failure_without_dims_keeps_provided_bytes_unknown() {
    let basis = sample_basis();
    let operator =
        Operator::new(IntegralFamily::OneElectron, OperatorKind::Kinetic).expect("valid operator");
    let options = WorkspaceQueryOptions::default();

    let failure = cintx::raw::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &[99, 1],
        None,
        &options,
    )
    .expect_err("invalid shell tuple should fail validation");

    assert!(matches!(
        failure.error,
        LibcintRsError::InvalidInput {
            field: "shell_tuple",
            ..
        }
    ));
    assert_eq!(failure.diagnostics.api, "raw.query_workspace");
    assert!(failure.diagnostics.dims.is_empty());
    assert!(failure.diagnostics.provided_bytes.is_none());
}

fn sample_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom should be valid");
    let shell_a = Shell::new(0, 1, vec![130.70932, 5.0331513], vec![0.154329, 0.535328])
        .expect("shell should be valid");
    let shell_b = Shell::new(1, 0, vec![3.42525091, 0.62391373], vec![0.154329, 0.535328])
        .expect("shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_a, shell_b]).expect("basis should be valid")
}
