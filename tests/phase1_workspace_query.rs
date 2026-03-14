use cintx::{
    Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, Shell,
    WorkspaceQueryOptions, query_workspace_raw, query_workspace_safe,
};

#[test]
fn deterministic_query_workspace() {
    let basis = sample_basis();
    let operator =
        Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap).expect("valid operator");
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: None,
        backend_candidate: "cpu",
        feature_flags: vec!["with-f12", "trace-workspace"],
    };
    let shell_tuple = [0, 1];

    let first = query_workspace_safe(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &options,
    )
    .expect("safe query should succeed");
    let second = query_workspace_safe(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &options,
    )
    .expect("identical query should stay deterministic");
    let raw = query_workspace_raw(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        Some(&first.natural_dims),
        &options,
    )
    .expect("raw-compatible query should validate natural dims");

    assert_eq!(first, second);
    assert_eq!(first, raw);
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
