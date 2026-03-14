use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, Shell,
    WorkspaceQueryOptions,
};

#[test]
fn no_partial_write_on_contract_error() {
    let basis = sample_basis();
    let options = WorkspaceQueryOptions::default();
    let one_electron_overlap = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");

    let mut undersized_real = vec![7.25; 17];
    let real_failure = cintx::safe::evaluate_into(
        &basis,
        one_electron_overlap,
        Representation::Cartesian,
        &[0, 1],
        &options,
        cintx::EvaluationOutputMut::Real(&mut undersized_real),
    )
    .expect_err("undersized real output must fail before execution");
    assert!(matches!(
        real_failure.error,
        LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: 18,
            got: 17,
        }
    ));
    assert_eq!(real_failure.diagnostics.api, "safe.evaluate_into");
    assert!(undersized_real.iter().all(|value| (*value - 7.25).abs() < f64::EPSILON));

    let mut undersized_spinor = vec![[11.0, -11.0]; 59];
    let spinor_failure = cintx::safe::evaluate_into(
        &basis,
        one_electron_overlap,
        Representation::Spinor,
        &[0, 1],
        &options,
        cintx::EvaluationOutputMut::Spinor(&mut undersized_spinor),
    )
    .expect_err("undersized spinor output must fail before execution");
    assert!(matches!(
        spinor_failure.error,
        LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: 60,
            got: 59,
        }
    ));
    assert_eq!(spinor_failure.diagnostics.api, "safe.evaluate_into");
    assert!(
        undersized_spinor
            .iter()
            .all(|value| (value[0] - 11.0).abs() < f64::EPSILON && (value[1] + 11.0).abs() < f64::EPSILON)
    );
}

fn sample_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");
    let shell_d = Shell::new(0, 2, vec![4.0, 1.0], vec![0.7, 0.3]).expect("d shell should be valid");
    let shell_p = Shell::new(1, 1, vec![3.0, 0.8], vec![0.6, 0.4]).expect("p shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_d, shell_p]).expect("basis should be valid")
}
