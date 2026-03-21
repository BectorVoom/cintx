#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    CpuRouteKey, EvaluationOutputMut, ExecutionRequest, IntegralFamily, LibcintRsError, Operator,
    OperatorKind, RawEvaluateRequest, RawQueryRequest, Representation, RouteStatus, RouteSurface,
    WorkspaceQueryOptions, raw, resolve_capi_route, resolve_raw_route, resolve_route,
    resolve_safe_route, route_manifest_entries, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{
    flatten_safe_output, phase2_cpu_options, phase3_optimizer_options, stable_raw_layout,
    stable_safe_basis,
};

const TOL_3C1E_ABS: f64 = 1e-7;
const TOL_3C1E_REL: f64 = 1e-5;
const TOL_3C2E_ABS: f64 = 1e-9;
const TOL_3C2E_REL: f64 = 1e-7;

const REDUCED_SAFE_SHLS: &[usize] = &[2, 0, 1];
const REDUCED_RAW_SHLS: &[i32] = &[2, 0, 1];
const ALT_RAW_SHLS_A: &[i32] = &[0, 1, 2];
const ALT_RAW_SHLS_B: &[i32] = &[2, 1, 0];
const ALT_RAW_SHLS_C: &[i32] = &[1, 3, 0];

const SUPPORTED_3C1E_REPRESENTATIONS: &[Representation] =
    &[Representation::Cartesian, Representation::Spherical];
const SUPPORTED_3C2E_REPRESENTATIONS: &[Representation] = &[
    Representation::Cartesian,
    Representation::Spherical,
    Representation::Spinor,
];

#[test]
fn three_c_route_inventory_is_complete_and_classified() {
    let three_c_entries = route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| {
            entry.key.family == IntegralFamily::ThreeCenterOneElectron
                || entry.key.family == IntegralFamily::ThreeCenterTwoElectron
        })
        .collect::<Vec<_>>();

    assert_eq!(
        three_c_entries.len(),
        7,
        "3c route inventory must include 5 implemented rows and 2 unsupported-policy rows"
    );

    let implemented = three_c_entries
        .iter()
        .filter(|entry| entry.status == RouteStatus::Implemented)
        .count();
    let unsupported = three_c_entries
        .iter()
        .filter(|entry| entry.status == RouteStatus::UnsupportedPolicy)
        .count();
    assert_eq!(
        implemented, 5,
        "3c inventory must keep 5 implemented routes"
    );
    assert_eq!(
        unsupported, 2,
        "3c inventory must keep two explicit unsupported-policy rows"
    );

    let route_ids = three_c_entries
        .iter()
        .map(|entry| entry.route_id)
        .collect::<std::collections::BTreeSet<_>>();
    for expected in [
        "int3c1e_kin.cart.cpu.direct.v1",
        "int3c1e_kin.sph.cpu.direct.v1",
        "int3c1e_kin.spinor.policy.unsupported.v1",
        "int3c2e_eri.cart.cpu.direct.v1",
        "int3c2e_eri.sph.cpu.direct.v1",
        "int3c2e_eri.spinor.cpu.direct.v1",
        "int3c2e_kin.spinor.policy.unsupported.v1",
    ] {
        assert!(
            route_ids.contains(expected),
            "missing 3c manifest route `{expected}`"
        );
    }
}

#[test]
fn three_c1e_safe_and_raw_match_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for &representation in SUPPORTED_3C1E_REPRESENTATIONS {
        let options = parity_options(representation, "three-c1e-safe-raw-wrapper");
        let operator = three_c1e_operator();

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!("safe evaluate failed for 3c1e {representation:?} reduced fixture: {err:?}")
        });
        let safe_scalars = flatten_safe_output(safe_tensor.output);

        let (wrapper_scalars, wrapper_dims) = wrapper_three_c(
            IntegralFamily::ThreeCenterOneElectron,
            &atm,
            &bas,
            &env,
            REDUCED_RAW_SHLS,
            representation,
        );

        assert_eq!(safe_tensor.dims, wrapper_dims);
        assert_eq!(safe_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &safe_scalars,
            TOL_3C1E_ABS,
            TOL_3C1E_REL,
            &format!("safe 3c1e {representation:?} reduced case vs wrapper"),
        );

        let (workspace, raw_scalars) = evaluate_raw_three_c(
            IntegralFamily::ThreeCenterOneElectron,
            &atm,
            &bas,
            &env,
            REDUCED_RAW_SHLS,
            representation,
            &options,
        );
        assert_eq!(workspace.dims, wrapper_dims);
        assert_eq!(raw_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            TOL_3C1E_ABS,
            TOL_3C1E_REL,
            &format!("raw 3c1e {representation:?} reduced case vs wrapper"),
        );
    }
}

#[test]
fn three_c2e_safe_and_raw_match_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for &representation in SUPPORTED_3C2E_REPRESENTATIONS {
        let options = parity_options(representation, "three-c2e-safe-raw-wrapper");
        let operator = three_c2e_operator();

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!("safe evaluate failed for 3c2e {representation:?} reduced fixture: {err:?}")
        });
        let safe_scalars = flatten_safe_output(safe_tensor.output);

        let (wrapper_scalars, wrapper_dims) = wrapper_three_c(
            IntegralFamily::ThreeCenterTwoElectron,
            &atm,
            &bas,
            &env,
            REDUCED_RAW_SHLS,
            representation,
        );

        assert_eq!(safe_tensor.dims, wrapper_dims);
        assert_eq!(safe_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &safe_scalars,
            TOL_3C2E_ABS,
            TOL_3C2E_REL,
            &format!("safe 3c2e {representation:?} reduced case vs wrapper"),
        );

        let (workspace, raw_scalars) = evaluate_raw_three_c(
            IntegralFamily::ThreeCenterTwoElectron,
            &atm,
            &bas,
            &env,
            REDUCED_RAW_SHLS,
            representation,
            &options,
        );
        assert_eq!(workspace.dims, wrapper_dims);
        assert_eq!(raw_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            TOL_3C2E_ABS,
            TOL_3C2E_REL,
            &format!("raw 3c2e {representation:?} reduced case vs wrapper"),
        );
    }
}

#[test]
fn three_c_safe_evaluate_into_matches_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for family in [
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ] {
        let operator = operator_for_family(family);
        let (abs_tol, rel_tol) = tolerances_for_family(family);

        for &representation in supported_representations_for_family(family) {
            let options = parity_options(representation, "three-c-safe-evaluate-into-wrapper");
            let (wrapper_scalars, wrapper_dims) =
                wrapper_three_c(family, &atm, &bas, &env, REDUCED_RAW_SHLS, representation);
            let element_count = checked_product(&wrapper_dims);

            match representation {
                Representation::Cartesian | Representation::Spherical => {
                    let mut output = vec![0.0f64; element_count];
                    let metadata = safe::evaluate_into(
                        &basis,
                        operator,
                        representation,
                        REDUCED_SAFE_SHLS,
                        &options,
                        EvaluationOutputMut::Real(&mut output),
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "safe evaluate_into failed for {family:?}/{representation:?} reduced fixture: {err:?}"
                        )
                    });
                    assert_eq!(metadata.dims, wrapper_dims);
                    assert_eq!(output.len(), wrapper_scalars.len());
                    assert_within_tolerance(
                        &wrapper_scalars,
                        &output,
                        abs_tol,
                        rel_tol,
                        &format!("safe evaluate_into {family:?}/{representation:?} vs wrapper"),
                    );
                }
                Representation::Spinor => {
                    let mut output = vec![[0.0f64; 2]; element_count];
                    let metadata = safe::evaluate_into(
                        &basis,
                        operator,
                        representation,
                        REDUCED_SAFE_SHLS,
                        &options,
                        EvaluationOutputMut::Spinor(&mut output),
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "safe evaluate_into failed for {family:?}/{representation:?} reduced fixture: {err:?}"
                        )
                    });
                    let flattened = flatten_spinor_pairs(output.as_slice());
                    assert_eq!(metadata.dims, wrapper_dims);
                    assert_eq!(flattened.len(), wrapper_scalars.len());
                    assert_within_tolerance(
                        &wrapper_scalars,
                        &flattened,
                        abs_tol,
                        rel_tol,
                        &format!("safe evaluate_into {family:?}/{representation:?} vs wrapper"),
                    );
                }
            }
        }
    }
}

#[test]
fn three_c_raw_layout_is_libcint_column_major_for_supported_representations() {
    let (atm, bas, env) = stable_raw_layout();

    for family in [
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ] {
        let (abs_tol, rel_tol) = tolerances_for_family(family);

        for &representation in supported_representations_for_family(family) {
            let options = parity_options(representation, "three-c-raw-layout");
            let (workspace, raw_scalars) = evaluate_raw_three_c(
                family,
                &atm,
                &bas,
                &env,
                REDUCED_RAW_SHLS,
                representation,
                &options,
            );
            let (wrapper_scalars, wrapper_dims) =
                wrapper_three_c(family, &atm, &bas, &env, REDUCED_RAW_SHLS, representation);

            assert_eq!(workspace.dims, wrapper_dims);
            let row_major_scalars = match representation {
                Representation::Cartesian | Representation::Spherical => {
                    col_major_to_row_major_real_rank3(&wrapper_scalars, &wrapper_dims)
                }
                Representation::Spinor => {
                    col_major_to_row_major_spinor_rank3(&wrapper_scalars, &wrapper_dims)
                }
            };
            assert_ne!(
                wrapper_scalars, row_major_scalars,
                "fixture must distinguish column-major and row-major ordering for {family:?}/{representation:?}"
            );
            assert_within_tolerance(
                &wrapper_scalars,
                &raw_scalars,
                abs_tol,
                rel_tol,
                &format!(
                    "raw {family:?}/{representation:?} should preserve libcint flat column-major layout"
                ),
            );
            let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
            assert!(
                row_major_max_diff > 1e-9,
                "raw {family:?}/{representation:?} unexpectedly matches row-major ordering: max_abs_diff={row_major_max_diff}",
            );
        }
    }
}

#[test]
fn three_c_raw_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();

    for family in [
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ] {
        let (abs_tol, rel_tol) = tolerances_for_family(family);

        for &representation in supported_representations_for_family(family) {
            let options = parity_options(representation, "three-c-raw-inputs");
            for shls in [
                ALT_RAW_SHLS_A,
                REDUCED_RAW_SHLS,
                ALT_RAW_SHLS_B,
                ALT_RAW_SHLS_C,
            ] {
                let (workspace, raw_scalars) =
                    evaluate_raw_three_c(family, &atm, &bas, &env, shls, representation, &options);
                let (wrapper_scalars, wrapper_dims) =
                    wrapper_three_c(family, &atm, &bas, &env, shls, representation);

                assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
                assert_within_tolerance(
                    &wrapper_scalars,
                    &raw_scalars,
                    abs_tol,
                    rel_tol,
                    &format!(
                        "raw {family:?}/{representation:?} shell interpretation shls={shls:?}"
                    ),
                );
            }
        }
    }
}

#[test]
fn three_c_raw_optimizer_mode_is_invariant_and_wrapper_aligned() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for family in [
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ] {
        let operator = operator_for_family(family);
        let (abs_tol, rel_tol) = tolerances_for_family(family);

        for &representation in supported_representations_for_family(family) {
            let baseline_options = phase3_optimizer_options(&[
                "three-c-optimizer-off",
                family_label(family),
                representation_label(representation),
            ]);
            let optimized_options = phase3_optimizer_options(&[
                "three-c-optimizer-on",
                family_label(family),
                representation_label(representation),
            ]);

            let safe_tensor = safe::evaluate(
                &basis,
                operator,
                representation,
                REDUCED_SAFE_SHLS,
                &baseline_options,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "safe evaluate failed for {family:?}/{representation:?} optimizer baseline: {err:?}"
                )
            });
            let safe_scalars = flatten_safe_output(safe_tensor.output);

            let baseline_workspace = raw::query_workspace_compat_with_sentinels(
                operator,
                representation,
                RawQueryRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: None,
                    cache: None,
                    opt: None,
                },
                &baseline_options,
            )
            .unwrap_or_else(|err| {
                panic!("baseline raw query failed for {family:?}/{representation:?}: {err:?}")
            });

            let optimizer_handle = wrapper_cint_from_raw_layout(&atm, &bas, &env, representation)
                .optimizer(optimizer_integral_for_family(family));
            let optimizer_ptr =
                NonNull::new(optimizer_handle.as_ptr() as *mut c_void).expect("optimizer pointer");
            let optimizer_query_cache = vec![0.0f64; baseline_workspace.cache_required_len];
            let optimized_workspace = raw::query_workspace_compat_with_sentinels(
                operator,
                representation,
                RawQueryRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: None,
                    cache: Some(optimizer_query_cache.as_slice()),
                    opt: Some(optimizer_ptr),
                },
                &optimized_options,
            )
            .unwrap_or_else(|err| {
                panic!("optimized raw query failed for {family:?}/{representation:?}: {err:?}")
            });

            let mut baseline_output = vec![0.0f64; baseline_workspace.required_bytes / 8];
            let mut optimized_output = vec![0.0f64; optimized_workspace.required_bytes / 8];
            let mut optimized_cache = vec![0.0f64; optimized_workspace.cache_required_len];

            raw::evaluate_compat(
                operator,
                representation,
                &baseline_workspace,
                RawEvaluateRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: &mut baseline_output,
                    cache: None,
                    opt: None,
                },
                &baseline_options,
            )
            .unwrap_or_else(|err| {
                panic!("baseline raw evaluate failed for {family:?}/{representation:?}: {err:?}")
            });

            raw::evaluate_compat(
                operator,
                representation,
                &optimized_workspace,
                RawEvaluateRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: &mut optimized_output,
                    cache: Some(optimized_cache.as_mut_slice()),
                    opt: Some(optimizer_ptr),
                },
                &optimized_options,
            )
            .unwrap_or_else(|err| {
                panic!("optimized raw evaluate failed for {family:?}/{representation:?}: {err:?}")
            });

            let (wrapper_scalars, wrapper_dims) =
                wrapper_three_c(family, &atm, &bas, &env, REDUCED_RAW_SHLS, representation);

            assert_eq!(safe_tensor.dims, baseline_workspace.dims);
            assert_eq!(safe_tensor.dims, optimized_workspace.dims);
            assert_eq!(safe_tensor.dims, wrapper_dims);
            assert_within_tolerance(
                &baseline_output,
                &optimized_output,
                abs_tol,
                rel_tol,
                &format!("raw {family:?}/{representation:?} optimizer on/off invariance"),
            );
            assert_within_tolerance(
                &safe_scalars,
                &baseline_output,
                abs_tol,
                rel_tol,
                &format!("safe/raw baseline parity {family:?}/{representation:?}"),
            );
            assert_within_tolerance(
                &safe_scalars,
                &optimized_output,
                abs_tol,
                rel_tol,
                &format!("safe/raw optimized parity {family:?}/{representation:?}"),
            );
            assert_within_tolerance(
                &wrapper_scalars,
                &baseline_output,
                abs_tol,
                rel_tol,
                &format!("wrapper/raw baseline parity {family:?}/{representation:?}"),
            );
            assert_within_tolerance(
                &wrapper_scalars,
                &optimized_output,
                abs_tol,
                rel_tol,
                &format!("wrapper/raw optimized parity {family:?}/{representation:?}"),
            );
        }
    }
}

#[test]
fn three_c_query_and_evaluate_surfaces_follow_shared_route_policy() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["three-c-route-policy"]);

    for family in [
        IntegralFamily::ThreeCenterOneElectron,
        IntegralFamily::ThreeCenterTwoElectron,
    ] {
        let operator = operator_for_family(family);

        for &representation in supported_representations_for_family(family) {
            let expected_route_id = route_id_for_supported_family(family, representation);
            let request =
                ExecutionRequest::from_safe(operator, representation, REDUCED_SAFE_SHLS, &options);

            let safe_route = resolve_safe_route(&request).unwrap_or_else(|err| {
                panic!("safe resolve failed for {family:?}/{representation:?}: {err:?}")
            });
            let raw_route = resolve_raw_route(&request).unwrap_or_else(|err| {
                panic!("raw resolve failed for {family:?}/{representation:?}: {err:?}")
            });
            let capi_route = resolve_capi_route(&request).unwrap_or_else(|err| {
                panic!("capi resolve failed for {family:?}/{representation:?}: {err:?}")
            });

            assert_eq!(safe_route.route_id, expected_route_id);
            assert_eq!(raw_route.route_id, expected_route_id);
            assert_eq!(capi_route.route_id, expected_route_id);

            let safe_query = safe::query_workspace(
                &basis,
                operator,
                representation,
                REDUCED_SAFE_SHLS,
                &options,
            )
            .unwrap_or_else(|err| {
                panic!("safe query failed for {family:?}/{representation:?}: {err:?}")
            });
            let safe_eval = safe::evaluate(
                &basis,
                operator,
                representation,
                REDUCED_SAFE_SHLS,
                &options,
            )
            .unwrap_or_else(|err| {
                panic!("safe evaluate failed for {family:?}/{representation:?}: {err:?}")
            });
            assert_eq!(safe_query.dims, safe_eval.dims);

            let raw_query = raw::query_workspace_compat_with_sentinels(
                operator,
                representation,
                RawQueryRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: None,
                    cache: None,
                    opt: None,
                },
                &options,
            )
            .unwrap_or_else(|err| {
                panic!("raw query failed for {family:?}/{representation:?}: {err:?}")
            });
            let mut raw_output = vec![0.0f64; raw_query.required_bytes / 8];
            let raw_result = raw::evaluate_compat(
                operator,
                representation,
                &raw_query,
                RawEvaluateRequest {
                    shls: REDUCED_RAW_SHLS,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: &mut raw_output,
                    cache: None,
                    opt: None,
                },
                &options,
            )
            .unwrap_or_else(|err| {
                panic!("raw evaluate failed for {family:?}/{representation:?}: {err:?}")
            });

            let route_key = CpuRouteKey::new(family, operator.kind(), representation);
            assert_eq!(route_key, safe_route.key);
            assert_eq!(route_key, raw_route.key);
            assert_eq!(raw_result.dims, raw_query.dims);
        }
    }
}

#[test]
fn three_c_unsupported_policy_route_is_blocked_by_shared_resolver() {
    assert!(
        Operator::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::Kinetic
        )
        .is_err(),
        "typed operator construction must reject unsupported 3c2e kinetic pair"
    );

    let unsupported_keys = [
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
    ];
    for unsupported_key in unsupported_keys {
        for surface in [RouteSurface::Safe, RouteSurface::Raw, RouteSurface::CAbi] {
            let err = resolve_route(unsupported_key, surface)
                .expect_err("unsupported 3c route should fail in shared resolver");
            assert!(
                matches!(
                    err,
                    LibcintRsError::UnsupportedApi {
                        api: "cpu.route",
                        ..
                    }
                ),
                "expected UnsupportedApi for unsupported 3c route {unsupported_key:?} on {surface:?}, got {err:?}",
            );
        }
    }
}

#[test]
fn reproduces_pre_fix_seeded_filler_mismatch_for_three_c2e_cartesian_case() {
    let (atm, bas, env) = stable_raw_layout();
    let (wrapper_scalars, wrapper_dims) = wrapper_three_c(
        IntegralFamily::ThreeCenterTwoElectron,
        &atm,
        &bas,
        &env,
        REDUCED_RAW_SHLS,
        Representation::Cartesian,
    );
    let legacy_scalars = legacy_seeded_real_scalars("int3c2e_ip1_cart", &wrapper_dims);

    let max_diff = max_abs_diff(&wrapper_scalars, &legacy_scalars);
    assert!(
        max_diff > 1e-3,
        "legacy seeded filler unexpectedly matched wrapper for reduced Cartesian 3c2e case",
    );
}

fn evaluate_raw_three_c(
    family: IntegralFamily,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
    options: &WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let operator = operator_for_family(family);
    let workspace = raw::query_workspace_compat_with_sentinels(
        operator,
        representation,
        RawQueryRequest {
            shls,
            dims: None,
            atm,
            bas,
            env,
            out: None,
            cache: None,
            opt: None,
        },
        options,
    )
    .unwrap_or_else(|err| {
        panic!("raw query failed for {family:?}/{representation:?} {shls:?}: {err:?}")
    });

    let mut output = vec![0.0f64; workspace.required_bytes / core::mem::size_of::<f64>()];
    raw::evaluate_compat(
        operator,
        representation,
        &workspace,
        RawEvaluateRequest {
            shls,
            dims: None,
            atm,
            bas,
            env,
            out: &mut output,
            cache: None,
            opt: None,
        },
        options,
    )
    .unwrap_or_else(|err| {
        panic!("raw evaluate failed for {family:?}/{representation:?} {shls:?}: {err:?}")
    });

    (workspace, output)
}

fn wrapper_three_c(
    family: IntegralFamily,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, representation);
    let shls_slice = shell_ranges_3(shls);
    let integral = match family {
        IntegralFamily::ThreeCenterOneElectron => "int3c1e_p2",
        IntegralFamily::ThreeCenterTwoElectron => "int3c2e_ip1",
        _ => panic!("wrapper_three_c only supports 3c families, got {family:?}"),
    };
    if family == IntegralFamily::ThreeCenterOneElectron && representation == Representation::Spinor
    {
        panic!(
            "3c1e spinor is policy-blocked: upstream libcint exits process in CINT3c1e_spinor_drv"
        );
    }

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let (values, dims) = cint.integrate(integral, None, shls_slice).into();
            (values, dims)
        }
        Representation::Spinor => {
            let (values, dims) = cint.integrate_spinor(integral, None, shls_slice).into();
            let mut flattened = Vec::with_capacity(values.len() * 2);
            for value in values {
                flattened.push(value.re);
                flattened.push(value.im);
            }
            (flattened, dims)
        }
    }
}

fn wrapper_cint_from_raw_layout(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    representation: Representation,
) -> CInt {
    let atm_rows = atm
        .chunks_exact(6)
        .map(|row| row.try_into().expect("atm rows must have 6 slots"))
        .collect();
    let bas_rows = bas
        .chunks_exact(8)
        .map(|row| row.try_into().expect("bas rows must have 8 slots"))
        .collect();
    let cint_type = match representation {
        Representation::Cartesian => CIntType::Cartesian,
        Representation::Spherical => CIntType::Spheric,
        Representation::Spinor => CIntType::Spinor,
    };

    CInt {
        atm: atm_rows,
        bas: bas_rows,
        ecpbas: Vec::new(),
        env: env.to_vec(),
        cint_type,
    }
}

fn shell_ranges_3(shls: &[i32]) -> [[usize; 2]; 3] {
    [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
        [shls[2] as usize, shls[2] as usize + 1],
    ]
}

fn parity_options(representation: Representation, gate: &'static str) -> WorkspaceQueryOptions {
    phase2_cpu_options(&[gate, representation_label(representation)])
}

fn representation_label(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "cart",
        Representation::Spherical => "sph",
        Representation::Spinor => "spinor",
    }
}

fn family_label(family: IntegralFamily) -> &'static str {
    match family {
        IntegralFamily::ThreeCenterOneElectron => "3c1e",
        IntegralFamily::ThreeCenterTwoElectron => "3c2e",
        _ => "other",
    }
}

fn operator_for_family(family: IntegralFamily) -> Operator {
    match family {
        IntegralFamily::ThreeCenterOneElectron => three_c1e_operator(),
        IntegralFamily::ThreeCenterTwoElectron => three_c2e_operator(),
        _ => panic!("unsupported family for 3c wrapper parity tests: {family:?}"),
    }
}

fn optimizer_integral_for_family(family: IntegralFamily) -> &'static str {
    match family {
        IntegralFamily::ThreeCenterOneElectron => "int3c1e_p2",
        IntegralFamily::ThreeCenterTwoElectron => "int3c2e_ip1",
        _ => "int3c2e_ip1",
    }
}

fn route_id_for_supported_family(
    family: IntegralFamily,
    representation: Representation,
) -> &'static str {
    match (family, representation) {
        (IntegralFamily::ThreeCenterOneElectron, Representation::Cartesian) => {
            "int3c1e_kin.cart.cpu.direct.v1"
        }
        (IntegralFamily::ThreeCenterOneElectron, Representation::Spherical) => {
            "int3c1e_kin.sph.cpu.direct.v1"
        }
        (IntegralFamily::ThreeCenterTwoElectron, Representation::Cartesian) => {
            "int3c2e_eri.cart.cpu.direct.v1"
        }
        (IntegralFamily::ThreeCenterTwoElectron, Representation::Spherical) => {
            "int3c2e_eri.sph.cpu.direct.v1"
        }
        (IntegralFamily::ThreeCenterTwoElectron, Representation::Spinor) => {
            "int3c2e_eri.spinor.cpu.direct.v1"
        }
        _ => panic!("unsupported route id lookup for {family:?}/{representation:?}"),
    }
}

fn three_c1e_operator() -> Operator {
    Operator::new(
        IntegralFamily::ThreeCenterOneElectron,
        OperatorKind::Kinetic,
    )
    .expect("3c1e kinetic operator should be valid")
}

fn three_c2e_operator() -> Operator {
    Operator::new(
        IntegralFamily::ThreeCenterTwoElectron,
        OperatorKind::ElectronRepulsion,
    )
    .expect("3c2e electron-repulsion operator should be valid")
}

fn supported_representations_for_family(family: IntegralFamily) -> &'static [Representation] {
    match family {
        IntegralFamily::ThreeCenterOneElectron => SUPPORTED_3C1E_REPRESENTATIONS,
        IntegralFamily::ThreeCenterTwoElectron => SUPPORTED_3C2E_REPRESENTATIONS,
        _ => &[],
    }
}

fn tolerances_for_family(family: IntegralFamily) -> (f64, f64) {
    match family {
        IntegralFamily::ThreeCenterOneElectron => (TOL_3C1E_ABS, TOL_3C1E_REL),
        IntegralFamily::ThreeCenterTwoElectron => (TOL_3C2E_ABS, TOL_3C2E_REL),
        _ => panic!("unsupported tolerance family: {family:?}"),
    }
}

fn assert_within_tolerance(
    expected: &[f64],
    actual: &[f64],
    abs_tol: f64,
    rel_tol: f64,
    context: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected and actual scalar lengths must match",
    );

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let abs_diff = (expected_value - actual_value).abs();
        if abs_diff <= abs_tol {
            continue;
        }

        let scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        let rel_diff = abs_diff / scale;
        assert!(
            rel_diff <= rel_tol,
            "{context}: mismatch at index {index}: expected={expected_value}, got={actual_value}, abs_diff={abs_diff}, rel_diff={rel_diff}",
        );
    }
}

fn flatten_spinor_pairs(values: &[[f64; 2]]) -> Vec<f64> {
    let mut flattened = Vec::with_capacity(values.len() * 2);
    for pair in values {
        flattened.push(pair[0]);
        flattened.push(pair[1]);
    }
    flattened
}

fn checked_product(dims: &[usize]) -> usize {
    let mut product = 1usize;
    for dim in dims {
        assert!(*dim > 0, "dimension values must be positive");
        product = product
            .checked_mul(*dim)
            .expect("dimension product should fit usize for reduced fixtures");
    }
    product
}

fn col_major_to_row_major_real_rank3(values: &[f64], dims: &[usize]) -> Vec<f64> {
    let mut row_major = vec![0.0f64; values.len()];
    match dims {
        [d0, d1, d2] => {
            for k in 0..*d2 {
                for j in 0..*d1 {
                    for i in 0..*d0 {
                        let col_index = i + d0 * (j + d1 * k);
                        let row_index = (i * d1 + j) * d2 + k;
                        row_major[row_index] = values[col_index];
                    }
                }
            }
        }
        [d0, d1, d2, d3] => {
            for m in 0..*d3 {
                for k in 0..*d2 {
                    for j in 0..*d1 {
                        for i in 0..*d0 {
                            let col_index = i + d0 * (j + d1 * (k + d2 * m));
                            let row_index = ((i * d1 + j) * d2 + k) * d3 + m;
                            row_major[row_index] = values[col_index];
                        }
                    }
                }
            }
        }
        _ => panic!("3c layouts must be rank-3 or rank-4, got {}", dims.len()),
    }

    row_major
}

fn col_major_to_row_major_spinor_rank3(values: &[f64], dims: &[usize]) -> Vec<f64> {
    assert!(values.len().is_multiple_of(2));
    let pairs = values
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect::<Vec<_>>();
    let mut row_pairs = vec![[0.0f64; 2]; pairs.len()];
    match dims {
        [d0, d1, d2] => {
            for k in 0..*d2 {
                for j in 0..*d1 {
                    for i in 0..*d0 {
                        let col_index = i + d0 * (j + d1 * k);
                        let row_index = (i * d1 + j) * d2 + k;
                        row_pairs[row_index] = pairs[col_index];
                    }
                }
            }
        }
        [d0, d1, d2, d3] => {
            for m in 0..*d3 {
                for k in 0..*d2 {
                    for j in 0..*d1 {
                        for i in 0..*d0 {
                            let col_index = i + d0 * (j + d1 * (k + d2 * m));
                            let row_index = ((i * d1 + j) * d2 + k) * d3 + m;
                            row_pairs[row_index] = pairs[col_index];
                        }
                    }
                }
            }
        }
        _ => panic!(
            "3c spinor layouts must be rank-3 or rank-4, got {}",
            dims.len()
        ),
    }

    flatten_spinor_pairs(row_pairs.as_slice())
}

fn legacy_seeded_real_scalars(route_symbol: &str, dims: &[usize]) -> Vec<f64> {
    let element_count = checked_product(dims);
    let mut output = vec![0.0f64; element_count];
    let seed = legacy_seed(route_symbol, dims);
    for (index, value) in output.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(17));
        *value = f64::from((raw % 4096) as u16) / 128.0;
    }
    output
}

fn legacy_seed(route_symbol: &str, dims: &[usize]) -> u64 {
    let mut seed = 0u64;
    for byte in route_symbol.bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
    }
    for dim in dims {
        let dim_u64 = u64::try_from(*dim).unwrap_or(u64::MAX);
        seed = seed.wrapping_mul(257).wrapping_add(dim_u64);
    }
    seed
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f64, f64::max)
}
