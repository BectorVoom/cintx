use cintx::{
    ALL_BOUND_SYMBOLS, CpuRouteKey, CpuRouteTarget, ExecutionBackend, ExecutionDispatch,
    ExecutionRequest, IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation,
    Spinor3c1eTransform, WorkspaceQueryOptions, route,
};
use std::collections::HashSet;

#[test]
fn cpu_backend_symbols_link() {
    for symbol in ALL_BOUND_SYMBOLS {
        assert!(
            !symbol.as_ptr().is_null(),
            "symbol `{}` should be linked",
            symbol.name()
        );
    }
}

#[test]
fn execution_request_contract() {
    let operator = Operator::new(
        IntegralFamily::ThreeCenterOneElectron,
        OperatorKind::Kinetic,
    )
    .expect("kinetic should be valid for 3c1e");
    let options = WorkspaceQueryOptions {
        memory_limit_bytes: Some(8 * 1024),
        backend_candidate: "cpu",
        feature_flags: vec!["trace-workspace", "phase2-contract"],
    };

    let safe_request =
        ExecutionRequest::from_safe(operator, Representation::Spinor, &[1, 4, 7], &options);
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
        vec!["trace-workspace".to_string(), "phase2-contract".to_string()]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingObligation {
    MustPassIn0206,
}

fn stable_family_required_matrix() -> Vec<(IntegralFamily, Representation, RoutingObligation)> {
    let families = [
        IntegralFamily::OneElectron,
        IntegralFamily::TwoElectron,
        IntegralFamily::TwoCenterTwoElectron,
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ];
    let representations = [
        Representation::Cartesian,
        Representation::Spherical,
        Representation::Spinor,
    ];

    let mut matrix = Vec::with_capacity(families.len() * representations.len());
    for family in families {
        for representation in representations {
            matrix.push((family, representation, RoutingObligation::MustPassIn0206));
        }
    }
    matrix
}

#[test]
fn stable_family_required_matrix_contract() {
    let matrix = stable_family_required_matrix();
    assert_eq!(
        matrix.len(),
        15,
        "stable-family matrix must stay complete: 5 families x 3 representations"
    );

    let mut unique_envelopes = HashSet::new();
    for &(family, representation, obligation) in &matrix {
        let inserted = unique_envelopes.insert((family, representation));
        assert!(
            inserted,
            "duplicate envelope in stable-family contract: {family:?} x {representation:?}"
        );
        assert_eq!(
            obligation,
            RoutingObligation::MustPassIn0206,
            "stable-family envelopes cannot be downgraded to unsupported before 02-06 routing"
        );
    }

    assert!(
        unique_envelopes.contains(&(
            IntegralFamily::ThreeCenterOneElectron,
            Representation::Spinor
        )),
        "3c1e spinor is mandatory in Phase 2 and must remain a required router target"
    );
}

#[test]
fn backend_route_matrix() {
    let matrix = [
        (
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Cartesian,
            true,
        ),
        (
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Spherical,
            true,
        ),
        (
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Spinor,
            true,
        ),
        (
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
            true,
        ),
        (
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
            true,
        ),
        (
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
            true,
        ),
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
            true,
        ),
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
            true,
        ),
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
            true,
        ),
        (
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Cartesian,
            true,
        ),
        (
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spherical,
            true,
        ),
        (
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
            false,
        ),
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
            true,
        ),
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
            true,
        ),
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
            true,
        ),
    ];

    for (family, operator, representation, should_route) in matrix {
        let key = CpuRouteKey::new(family, operator, representation);
        match (route(key), should_route) {
            (Ok(route_target), true) => {
                let symbol = route_target.entry_symbol();
                assert_eq!(symbol.family(), family);
                assert_eq!(symbol.operator(), operator);
                assert_eq!(symbol.representation(), representation);
                assert!(!symbol.as_ptr().is_null());
            }
            (Err(LibcintRsError::UnsupportedApi { api, .. }), false) => {
                assert_eq!(api, "cpu.route");
            }
            (result, _) => panic!(
                "unexpected route result for {family:?}/{operator:?}/{representation:?}: {result:?}"
            ),
        }
    }
}

#[test]
fn three_c_one_e_spinor_supported() {
    let route_target = route(CpuRouteKey::new(
        IntegralFamily::ThreeCenterOneElectron,
        OperatorKind::Kinetic,
        Representation::Spinor,
    ))
    .expect("3c1e spinor must be routable through dedicated adapter path");

    match route_target {
        CpuRouteTarget::ThreeCenterOneElectronSpinor(adapter) => {
            assert_eq!(
                adapter.transform,
                Spinor3c1eTransform::SphericalKernelToSpinorLayout
            );
            assert_eq!(adapter.driver_symbol.name(), "int3c1e_p2_sph");
            assert!(!adapter.driver_symbol.as_ptr().is_null());
        }
        other => panic!("expected spinor adapter route, got {other:?}"),
    }
}
