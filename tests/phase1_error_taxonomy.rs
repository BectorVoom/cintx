use cintx::{Atom, IntegralFamily, LibcintRsError, Operator, OperatorKind, Shell, validate_dims};

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
