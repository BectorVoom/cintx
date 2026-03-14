use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, Shell,
};

#[test]
fn typed_model_construction() {
    let oxygen = Atom::new(8, [0.0, 0.0, 0.1173]).expect("valid atom should build");
    let shell = Shell::new(0, 1, vec![130.70932, 5.0331513], vec![0.154329, 0.535328])
        .expect("valid shell should build");
    let basis = BasisSet::new(vec![oxygen.clone()], vec![shell.clone()])
        .expect("basis should reference existing atom indices");
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("overlap operator should be valid for one-electron family");

    assert_eq!(oxygen.atomic_number(), 8);
    assert_eq!(shell.primitives().len(), 2);
    assert_eq!(basis.atoms().len(), 1);
    assert_eq!(basis.shells().len(), 1);
    assert_eq!(operator.kind(), OperatorKind::Overlap);
    assert_eq!(Representation::Spinor.as_str(), "spinor");

    assert!(Atom::new(0, [0.0, 0.0, 0.0]).is_err());
    assert!(Shell::new(0, 1, vec![1.0], vec![]).is_err());
    assert!(Operator::new(IntegralFamily::OneElectron, OperatorKind::ElectronRepulsion).is_err());
}

#[test]
fn safe01_basis_rejects_shell_outside_atom_range() {
    let atom = Atom::new(1, [0.0, 0.0, 0.0]).expect("atom should build");
    let shell = Shell::new(1, 0, vec![1.0], vec![1.0]).expect("shell metadata should build");

    let error = BasisSet::new(vec![atom], vec![shell]).expect_err("shell index must be in bounds");
    assert!(matches!(
        error,
        LibcintRsError::InvalidInput {
            field: "shell.center_index",
            ..
        }
    ));
}

#[test]
fn safe01_shell_constructor_invariants() {
    let non_positive_exponent = Shell::new(0, 0, vec![0.0], vec![1.0]);
    assert!(matches!(
        non_positive_exponent,
        Err(LibcintRsError::InvalidInput {
            field: "exponent",
            ..
        })
    ));

    let all_zero_coefficients = Shell::new(0, 0, vec![1.0, 2.0], vec![0.0, 0.0]);
    assert!(matches!(
        all_zero_coefficients,
        Err(LibcintRsError::InvalidInput {
            field: "coefficients",
            ..
        })
    ));

    let unsupported_angular_momentum = Shell::new(0, 9, vec![1.0], vec![1.0]);
    assert!(matches!(
        unsupported_angular_momentum,
        Err(LibcintRsError::UnsupportedApi { api: "shell", .. })
    ));
}
