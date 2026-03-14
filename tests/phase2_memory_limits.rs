use cintx::{
    Atom, BasisSet, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, Shell,
    WorkspaceQueryOptions,
};

#[test]
fn allocation_paths_use_fallible_policy() {
    let real_error = cintx::runtime::memory::allocator::try_alloc_real_buffer(
        usize::MAX,
        "test.real_allocation",
    )
    .expect_err("oversized real allocation must surface typed failure");
    assert!(matches!(
        real_error,
        LibcintRsError::AllocationFailure {
            operation: "test.real_allocation",
            ..
        }
    ));

    let spinor_error = cintx::runtime::memory::allocator::try_alloc_spinor_buffer(
        usize::MAX,
        "test.spinor_allocation",
    )
    .expect_err("oversized spinor allocation must surface typed failure");
    assert!(matches!(
        spinor_error,
        LibcintRsError::AllocationFailure {
            operation: "test.spinor_allocation",
            ..
        }
    ));
}

#[test]
fn chunk_or_memory_limit_exceeded() {
    let basis = sample_basis();
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");
    let shell_tuple = [0, 1];

    let baseline_options = WorkspaceQueryOptions::default();
    let chunk_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(384),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy"],
    };
    let exceeded_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(320),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy"],
    };

    let baseline_query = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &baseline_options,
    )
    .expect("baseline query should succeed");
    let chunk_query = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &chunk_options,
    )
    .expect("chunk-feasible memory limit should still query successfully");

    assert_eq!(baseline_query, chunk_query);
    assert_eq!(chunk_query.element_count, 15);
    assert_eq!(chunk_query.scratch_bytes, 352);
    assert_eq!(chunk_query.required_bytes, 512);
    assert!(
        chunk_query.required_bytes > chunk_options.memory_limit_bytes.expect("chunk limit present")
    );

    let mut baseline_output = vec![0.0; chunk_query.element_count];
    let baseline_meta = cintx::safe::evaluate_into(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &baseline_options,
        cintx::EvaluationOutputMut::Real(&mut baseline_output),
    )
    .expect("baseline execute should succeed");

    let mut chunked_output = vec![0.0; chunk_query.element_count];
    let chunk_meta = cintx::safe::evaluate_into(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &chunk_options,
        cintx::EvaluationOutputMut::Real(&mut chunked_output),
    )
    .expect("chunk-feasible execute should succeed");

    assert_eq!(baseline_meta, chunk_meta);
    assert_eq!(chunked_output, baseline_output);

    let query_failure = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &exceeded_options,
    )
    .expect_err("infeasible limit must fail during query");
    assert!(matches!(
        query_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 512,
            limit_bytes: 320,
        }
    ));

    let mut untouched_output = vec![13.0; chunk_query.element_count];
    let execute_failure = cintx::safe::evaluate_into(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &exceeded_options,
        cintx::EvaluationOutputMut::Real(&mut untouched_output),
    )
    .expect_err("infeasible limit must fail during execute");
    assert!(matches!(
        execute_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 512,
            limit_bytes: 320,
        }
    ));
    assert!(untouched_output.iter().all(|value| (*value - 13.0).abs() < f64::EPSILON));
}

fn sample_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");
    let shell_d = Shell::new(0, 2, vec![4.0, 1.0], vec![0.7, 0.3]).expect("d shell should be valid");
    let shell_p = Shell::new(1, 1, vec![3.0, 0.8], vec![0.6, 0.4]).expect("p shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_d, shell_p]).expect("basis should be valid")
}
