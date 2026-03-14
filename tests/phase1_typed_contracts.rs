use cintx::{Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, Shell};

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
