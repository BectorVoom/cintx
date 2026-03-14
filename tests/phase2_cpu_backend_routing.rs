use cintx::{
    ExecutionBackend, ExecutionDispatch, ExecutionRequest, IntegralFamily, Operator, OperatorKind,
    Representation, WorkspaceQueryOptions,
};

#[link(name = "cint_phase2_cpu", kind = "static")]
#[link(name = "m")]
unsafe extern "C" {
    fn int1e_ovlp_cart();
    fn int1e_ovlp_sph();
    fn int1e_ovlp_spinor();
    fn int2e_cart();
    fn int2e_sph();
    fn int2e_spinor();
    fn int2c2e_ip1_cart();
    fn int2c2e_ip1_sph();
    fn int2c2e_ip1_spinor();
    fn int3c1e_p2_cart();
    fn int3c1e_p2_sph();
    fn int3c1e_p2_spinor();
    fn int3c2e_ip1_cart();
    fn int3c2e_ip1_sph();
    fn int3c2e_ip1_spinor();
}

#[test]
fn cpu_backend_symbols_link() {
    let symbols: &[(&str, *const ())] = &[
        ("int1e_ovlp_cart", int1e_ovlp_cart as *const ()),
        ("int1e_ovlp_sph", int1e_ovlp_sph as *const ()),
        ("int1e_ovlp_spinor", int1e_ovlp_spinor as *const ()),
        ("int2e_cart", int2e_cart as *const ()),
        ("int2e_sph", int2e_sph as *const ()),
        ("int2e_spinor", int2e_spinor as *const ()),
        ("int2c2e_ip1_cart", int2c2e_ip1_cart as *const ()),
        ("int2c2e_ip1_sph", int2c2e_ip1_sph as *const ()),
        ("int2c2e_ip1_spinor", int2c2e_ip1_spinor as *const ()),
        ("int3c1e_p2_cart", int3c1e_p2_cart as *const ()),
        ("int3c1e_p2_sph", int3c1e_p2_sph as *const ()),
        ("int3c1e_p2_spinor", int3c1e_p2_spinor as *const ()),
        ("int3c2e_ip1_cart", int3c2e_ip1_cart as *const ()),
        ("int3c2e_ip1_sph", int3c2e_ip1_sph as *const ()),
        ("int3c2e_ip1_spinor", int3c2e_ip1_spinor as *const ()),
    ];

    for (name, symbol) in symbols {
        assert!(!symbol.is_null(), "symbol `{name}` should be linked");
    }
}

#[test]
fn execution_request_contract() {
    let operator = Operator::new(IntegralFamily::ThreeCenterOneElectron, OperatorKind::Kinetic)
        .expect("kinetic should be valid for 3c1e");
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(8 * 1024),
        backend_candidate: "cpu",
        feature_flags: vec!["trace-workspace", "phase2-contract"],
    };

    let safe_request = ExecutionRequest::from_safe(
        operator,
        Representation::Spinor,
        &[1, 4, 7],
        &options,
    );
    let raw_request = ExecutionRequest::from_raw(
        operator,
        Representation::Spinor,
        &[1, 4, 7],
        Some(&[2, 3, 4]),
        &options,
    );

    assert_eq!(
        safe_request.operator.family,
        IntegralFamily::ThreeCenterOneElectron
    );
    assert_eq!(safe_request.operator.kind, OperatorKind::Kinetic);
    assert_eq!(safe_request.representation, Representation::Spinor);
    assert_eq!(safe_request.shell_tuple, vec![1, 4, 7]);
    assert_eq!(safe_request.dims, None);
    assert_eq!(safe_request.memory.memory_limit_bytes, Some(8 * 1024));
    assert_eq!(safe_request.memory.backend_candidate, "cpu");
    assert_eq!(
        safe_request.memory.feature_flags,
        vec![
            "trace-workspace".to_string(),
            "phase2-contract".to_string()
        ]
    );

    assert_eq!(raw_request.operator, safe_request.operator);
    assert_eq!(raw_request.representation, safe_request.representation);
    assert_eq!(raw_request.shell_tuple, safe_request.shell_tuple);
    assert_eq!(raw_request.dims, Some(vec![2, 3, 4]));
    assert_eq!(raw_request.memory, safe_request.memory);

    let dispatch = ExecutionDispatch::cpu(raw_request.clone());
    assert_eq!(dispatch.backend, ExecutionBackend::CpuReference);
    assert_eq!(dispatch.backend.as_str(), "cpu-reference");
    assert_eq!(dispatch.request, raw_request);
}
