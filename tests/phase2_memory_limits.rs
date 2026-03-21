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
        chunk_query.required_bytes
            > chunk_options
                .memory_limit_bytes
                .expect("chunk limit present")
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
    assert!(
        untouched_output
            .iter()
            .all(|value| (*value - 13.0).abs() < f64::EPSILON)
    );
}

#[test]
fn spinor_chunked_execute_matches_unlimited_execution() {
    let basis = sample_basis();
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");
    let shell_tuple = [0, 1];

    let unlimited_options = WorkspaceQueryOptions::default();
    let chunked_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(768),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy"],
    };

    let chunk_query = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spinor,
        &shell_tuple,
        &chunked_options,
    )
    .expect("spinor query should succeed under chunk-feasible limit");
    assert_eq!(chunk_query.required_bytes, 1344);
    assert!(chunk_query.required_bytes > 768);

    let baseline_tensor = cintx::safe::evaluate(
        &basis,
        operator,
        Representation::Spinor,
        &shell_tuple,
        &unlimited_options,
    )
    .expect("unlimited spinor evaluate should succeed");
    let chunked_tensor = cintx::safe::evaluate(
        &basis,
        operator,
        Representation::Spinor,
        &shell_tuple,
        &chunked_options,
    )
    .expect("chunked spinor evaluate should succeed");

    assert_eq!(chunked_tensor.dims, baseline_tensor.dims);
    assert_eq!(chunked_tensor.dims, vec![10, 6]);

    let baseline_values = match baseline_tensor.output {
        cintx::EvaluationOutput::Spinor(values) => values,
        other => panic!("expected spinor output from spinor evaluate, got {other:?}"),
    };
    let chunked_values = match chunked_tensor.output {
        cintx::EvaluationOutput::Spinor(values) => values,
        other => panic!("expected spinor output from spinor evaluate, got {other:?}"),
    };
    assert_eq!(chunked_values, baseline_values);
}

#[test]
fn chunk_feasibility_boundary_is_explicit() {
    let basis = sample_basis();
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");
    let shell_tuple = [0, 1];

    let just_infeasible = WorkspaceQueryOptions {
        memory_limit_bytes: Some(383),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy"],
    };
    let minimal_feasible = WorkspaceQueryOptions {
        memory_limit_bytes: Some(384),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy"],
    };

    let infeasible = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &just_infeasible,
    )
    .expect_err("limit below chunk minimum must fail");
    assert!(matches!(
        infeasible.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 512,
            limit_bytes: 383,
        }
    ));

    let feasible = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &minimal_feasible,
    )
    .expect("minimum chunk-feasible limit should succeed");
    assert_eq!(feasible.required_bytes, 512);
}

#[test]
fn api_memory_policy_threading() {
    let basis = sample_basis();
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");
    let shell_tuple = [0, 1];

    let feasible_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(384),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy-threading"],
    };
    let infeasible_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(320),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-policy-threading"],
    };

    let safe_query = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &feasible_options,
    )
    .expect("safe query should honor feasible memory policy");
    let raw_query = cintx::raw::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        None,
        &feasible_options,
    )
    .expect("raw query should honor feasible memory policy");
    assert_eq!(raw_query, safe_query);

    let safe_failure = cintx::safe::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        &infeasible_options,
    )
    .expect_err("safe query should fail under infeasible limit");
    let raw_failure = cintx::raw::query_workspace(
        &basis,
        operator,
        Representation::Spherical,
        &shell_tuple,
        None,
        &infeasible_options,
    )
    .expect_err("raw query should fail under infeasible limit");

    assert!(matches!(
        safe_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 512,
            limit_bytes: 320,
        }
    ));
    assert!(matches!(
        raw_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 512,
            limit_bytes: 320,
        }
    ));
    assert_eq!(safe_failure.diagnostics.required_bytes, Some(512));
    assert_eq!(raw_failure.diagnostics.required_bytes, Some(512));
}

#[test]
fn raw_query_execute_memory_contract() {
    let (atm, bas, env) = raw_sample_layout();
    let operator = raw_one_electron_overlap();
    let feasible_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(320),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-contract"],
    };
    let infeasible_options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(319),
        backend_candidate: "cpu",
        feature_flags: vec!["phase2-memory-contract"],
    };

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
        &feasible_options,
    )
    .expect("query should succeed under feasible memory cap");
    assert_eq!(queried.required_bytes, 24);
    assert_eq!(queried.memory_required_bytes, 320);
    assert_eq!(queried.memory_working_set_bytes, 320);
    assert_eq!(queried.memory_scratch_bytes, 288);
    assert_eq!(queried.chunk_elements, queried.required_elements);
    assert_eq!(queried.chunk_count, 1);

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![0.0f64; required_scalars + 2];
    output[required_scalars] = 77.0;
    output[required_scalars + 1] = 99.0;
    let result = cintx::raw::evaluate_compat(
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
        &feasible_options,
    )
    .expect("execute should succeed when query and execute memory policies agree");
    assert_eq!(result.required_bytes, queried.required_bytes);
    assert_eq!(output[required_scalars], 77.0);
    assert_eq!(output[required_scalars + 1], 99.0);

    let query_failure = cintx::raw::query_workspace_compat_with_sentinels(
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
        &infeasible_options,
    )
    .expect_err("query should fail for infeasible memory limit");
    assert!(matches!(
        query_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 320,
            limit_bytes: 319,
        }
    ));

    let mut output = vec![0.0f64; required_scalars];
    let execute_failure = cintx::raw::evaluate_compat(
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
        &infeasible_options,
    )
    .expect_err("execute should fail when memory policy becomes infeasible");
    assert!(matches!(
        execute_failure.error,
        LibcintRsError::MemoryLimitExceeded {
            required_bytes: 320,
            limit_bytes: 319,
        }
    ));
}

fn sample_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");
    let shell_d =
        Shell::new(0, 2, vec![4.0, 1.0], vec![0.7, 0.3]).expect("d shell should be valid");
    let shell_p =
        Shell::new(1, 1, vec![3.0, 0.8], vec![0.6, 0.4]).expect("p shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_d, shell_p]).expect("basis should be valid")
}

fn raw_sample_layout() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

fn raw_one_electron_overlap() -> Operator {
    Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap operator should be valid")
}
