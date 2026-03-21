use cintx::{
    ALL_BOUND_SYMBOLS, CpuRouteKey, ExecutionBackend, ExecutionDispatch, ExecutionRequest,
    IntegralFamily, LibcintRsError, Operator, OperatorKind, Representation, WorkspaceQueryOptions,
    route,
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
        feature_flags: vec!["trace-workspace", "phase2-contract", "trace-workspace"],
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
    let expected_normalized_feature_flags = vec!["phase2-contract", "trace-workspace"];
    assert_eq!(
        options.normalized_feature_flags(),
        expected_normalized_feature_flags,
        "query options must canonicalize feature flags before request construction"
    );
    let expected_feature_flags = expected_normalized_feature_flags
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert_eq!(safe_request.memory.feature_flags, expected_feature_flags);

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

    let mut matrix = Vec::with_capacity(14);
    for family in families {
        for representation in representations {
            if family == IntegralFamily::ThreeCenterOneElectron
                && representation == Representation::Spinor
            {
                continue;
            }
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
        14,
        "stable-family matrix must include all currently supported envelopes"
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
        !unique_envelopes.contains(&(
            IntegralFamily::ThreeCenterOneElectron,
            Representation::Spinor
        )),
        "3c1e spinor must remain outside the supported route envelope until Rust-native transform implementation is complete"
    );
}

fn stable_family_route_keys() -> Vec<CpuRouteKey> {
    let mut keys = Vec::with_capacity(14);
    for representation in [
        Representation::Cartesian,
        Representation::Spherical,
        Representation::Spinor,
    ] {
        keys.push(CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            representation,
        ));
        keys.push(CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        ));
        keys.push(CpuRouteKey::new(
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        ));
        if representation != Representation::Spinor {
            keys.push(CpuRouteKey::new(
                IntegralFamily::ThreeCenterOneElectron,
                OperatorKind::Kinetic,
                representation,
            ));
        }
        keys.push(CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        ));
    }
    keys
}

#[test]
fn backend_route_matrix() {
    for key in stable_family_route_keys() {
        let symbol = route(key)
            .expect("all stable-family envelopes must route in phase 2 backend map")
            .entry_symbol();
        assert_eq!(symbol.family(), key.family);
        assert_eq!(symbol.operator(), key.operator);
        assert_eq!(symbol.representation(), key.representation);
        assert!(!symbol.as_ptr().is_null());
    }
}

#[test]
fn three_c_one_e_spinor_is_policy_blocked() {
    let err = route(CpuRouteKey::new(
        IntegralFamily::ThreeCenterOneElectron,
        OperatorKind::Kinetic,
        Representation::Spinor,
    ))
    .expect_err("3c1e spinor must be blocked by shared route policy");
    assert!(matches!(
        err,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            ..
        }
    ));
}

#[test]
fn stable_family_route_matrix_complete() {
    let matrix = stable_family_route_keys();
    assert_eq!(
        matrix.len(),
        14,
        "stable-family matrix must include all currently supported envelopes"
    );

    let mut unique_envelopes = HashSet::new();
    for key in matrix {
        unique_envelopes.insert((key.family, key.representation));
        let symbol = route(key)
            .unwrap_or_else(|err| {
                panic!("stable-family route unexpectedly unsupported: {key:?} -> {err:?}")
            })
            .entry_symbol();
        assert_eq!(symbol.family(), key.family);
        assert_eq!(symbol.operator(), key.operator);
        assert!(!symbol.as_ptr().is_null());
    }

    assert_eq!(unique_envelopes.len(), 14);

    let out_of_scope = [
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::Overlap,
            Representation::Spherical,
        ),
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
    ];
    for key in out_of_scope {
        let err = route(key).expect_err("out-of-scope route should fail with typed unsupported");
        assert!(
            matches!(
                err,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "expected typed UnsupportedApi for out-of-scope key {key:?}, got {err:?}"
        );
    }
}
