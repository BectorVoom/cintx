use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Shell, validate_dims,
};

#[test]
fn typed_error_categories() {
    let invalid_atom = Atom::new(0, [0.0, 0.0, 0.0]);
    assert!(matches!(
        invalid_atom,
        Err(LibcintRsError::InvalidInput {
            field: "atomic_number",
            ..
        })
    ));

    let layout_error = Shell::new(0, 0, vec![1.0, 0.5], vec![1.0]);
    assert!(matches!(
        layout_error,
        Err(LibcintRsError::InvalidLayout {
            item: "primitive coefficients",
            expected: 2,
            got: 1,
        })
    ));

    let unsupported_operator =
        Operator::new(IntegralFamily::OneElectron, OperatorKind::ElectronRepulsion);
    assert!(matches!(
        unsupported_operator,
        Err(LibcintRsError::UnsupportedApi {
            api: "operator",
            ..
        })
    ));

    let dims_error = validate_dims(&[3, 4, 5], &[3, 4, 6]);
    assert!(matches!(
        dims_error,
        Err(LibcintRsError::DimsBufferMismatch { .. })
    ));

    let unsupported_representation = LibcintRsError::UnsupportedRepresentation {
        api: "int2e",
        representation: "spinor",
    };
    assert!(matches!(
        unsupported_representation,
        LibcintRsError::UnsupportedRepresentation { .. }
    ));

    let memory_limit_error = LibcintRsError::MemoryLimitExceeded {
        required_bytes: 2048,
        limit_bytes: 1024,
    };
    assert!(matches!(
        memory_limit_error,
        LibcintRsError::MemoryLimitExceeded { .. }
    ));

    let allocation_failure = LibcintRsError::AllocationFailure {
        operation: "workspace_allocation",
        detail: "allocator returned null".to_string(),
    };
    assert!(matches!(
        allocation_failure,
        LibcintRsError::AllocationFailure { .. }
    ));

    let backend_failure = LibcintRsError::BackendFailure {
        backend: "cpu",
        detail: "kernel dispatch failed".to_string(),
    };
    assert!(matches!(
        backend_failure,
        LibcintRsError::BackendFailure { .. }
    ));
}

#[test]
fn typed_error_constructor_failures_are_stable() {
    let invalid_coordinates = Atom::new(8, [f64::NAN, 0.0, 0.0]);
    assert!(matches!(
        invalid_coordinates,
        Err(LibcintRsError::InvalidInput {
            field: "coordinates",
            ..
        })
    ));

    let mismatched_contraction = Shell::new(0, 1, vec![1.0], vec![1.0, 0.5]);
    assert!(matches!(
        mismatched_contraction,
        Err(LibcintRsError::InvalidLayout {
            item: "primitive coefficients",
            expected: 1,
            got: 2,
        })
    ));

    let atom = Atom::new(1, [0.0, 0.0, 0.0]).expect("atom should build");
    let shell = Shell::new(2, 0, vec![1.0], vec![1.0]).expect("shell should build");
    let basis_error = BasisSet::new(vec![atom], vec![shell]);
    assert!(matches!(
        basis_error,
        Err(LibcintRsError::InvalidInput {
            field: "shell.center_index",
            ..
        })
    ));

    let dims_error = validate_dims(&[2, 2], &[2, 3]);
    assert!(matches!(
        dims_error,
        Err(LibcintRsError::DimsBufferMismatch {
            expected,
            provided,
        }) if expected == vec![2, 2] && provided == vec![2, 3]
    ));
}
