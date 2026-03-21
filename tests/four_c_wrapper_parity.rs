#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    CpuRouteKey, EvaluationOutputMut, ExecutionRequest, IntegralFamily, LibcintRsError, Operator,
    OperatorKind, RawEvaluateRequest, RawQueryRequest, Representation, RouteOptimizerMode,
    RouteStatus, RouteSurface, WorkspaceQueryOptions, raw, resolve_capi_route, resolve_raw_route,
    resolve_route, resolve_safe_route, route_manifest_entries, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{phase2_cpu_options, stable_raw_layout, stable_safe_basis};

const TOL_4C_ABS: f64 = 1e-6;
const TOL_4C_REL: f64 = 1e-5;

const REDUCED_SAFE_SHLS: &[usize] = &[0, 1, 2, 3];
const REDUCED_RAW_SHLS: &[i32] = &[0, 1, 2, 3];
const ALT_RAW_SHLS_A: &[i32] = &[3, 2, 1, 0];
const ALT_RAW_SHLS_B: &[i32] = &[2, 0, 3, 1];

const SUPPORTED_REPRESENTATIONS: &[Representation] =
    &[Representation::Cartesian, Representation::Spherical];

#[test]
fn four_c_route_inventory_is_complete_and_classified() {
    let four_c_entries = route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.key.family == IntegralFamily::FourCenterOneElectron)
        .collect::<Vec<_>>();

    assert_eq!(
        four_c_entries.len(),
        3,
        "4c route inventory must include two implemented rows and one unsupported-policy row",
    );

    let implemented = four_c_entries
        .iter()
        .filter(|entry| entry.status == RouteStatus::Implemented)
        .count();
    let unsupported = four_c_entries
        .iter()
        .filter(|entry| entry.status == RouteStatus::UnsupportedPolicy)
        .count();

    assert_eq!(
        implemented, 2,
        "4c inventory must keep two implemented routes"
    );
    assert_eq!(
        unsupported, 1,
        "4c inventory must keep one explicit unsupported-policy row",
    );

    let route_ids = four_c_entries
        .iter()
        .map(|entry| entry.route_id)
        .collect::<std::collections::BTreeSet<_>>();
    for expected in [
        "int4c1e_eri.cart.cpu.direct.v1",
        "int4c1e_eri.sph.cpu.direct.v1",
        "int4c1e_eri.spinor.policy.unsupported.v1",
    ] {
        assert!(
            route_ids.contains(expected),
            "missing 4c manifest route `{expected}`",
        );
    }
}

#[test]
fn four_c_safe_and_raw_match_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-safe-raw-wrapper");
        let operator = four_c_operator();

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!("safe evaluate failed for 4c {representation:?} reduced fixture: {err:?}")
        });

        let (wrapper_scalars, wrapper_dims) =
            wrapper_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(safe_tensor.dims, wrapper_dims);
        let safe_scalars = match safe_tensor.output {
            cintx::EvaluationOutput::Real(values) => values,
            cintx::EvaluationOutput::Spinor(_) => panic!("4c routes must be real-valued"),
        };
        assert_eq!(safe_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &safe_scalars,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("safe 4c {representation:?} reduced case vs wrapper"),
        );

        let (workspace, raw_scalars) =
            evaluate_raw_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation, &options);
        assert_eq!(workspace.dims, wrapper_dims);
        assert_eq!(raw_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("raw 4c {representation:?} reduced case vs wrapper"),
        );
    }
}

#[test]
fn four_c_safe_evaluate_into_matches_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let operator = four_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-safe-evaluate-into-wrapper");
        let (wrapper_scalars, wrapper_dims) =
            wrapper_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);
        let element_count = checked_product(&wrapper_dims);

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
            panic!("safe evaluate_into failed for 4c {representation:?}: {err:?}")
        });

        assert_eq!(metadata.dims, wrapper_dims);
        assert_eq!(output.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &output,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("safe evaluate_into 4c {representation:?} vs wrapper"),
        );
    }
}

#[test]
fn four_c_raw_uses_libcint_column_major_layout() {
    let (atm, bas, env) = stable_raw_layout();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-raw-layout");
        let (workspace, raw_scalars) =
            evaluate_raw_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation, &options);
        let (wrapper_scalars, wrapper_dims) =
            wrapper_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(workspace.dims, wrapper_dims);
        let row_major_scalars = col_major_to_row_major_real_rank4(&wrapper_scalars, &wrapper_dims);
        assert_ne!(
            wrapper_scalars, row_major_scalars,
            "fixture must distinguish column-major and row-major ordering for {representation:?}",
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("raw 4c {representation:?} should preserve libcint flat column-major layout"),
        );
        let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
        assert!(
            row_major_max_diff > 1e-9,
            "raw 4c {representation:?} unexpectedly matches row-major ordering: max_abs_diff={row_major_max_diff}",
        );
    }
}

#[test]
fn four_c_raw_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-raw-inputs");
        for shls in [REDUCED_RAW_SHLS, ALT_RAW_SHLS_A, ALT_RAW_SHLS_B] {
            let (workspace, raw_scalars) =
                evaluate_raw_four_c(&atm, &bas, &env, shls, representation, &options);
            let (wrapper_scalars, wrapper_dims) =
                wrapper_four_c(&atm, &bas, &env, shls, representation);

            assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
            assert_within_tolerance(
                &wrapper_scalars,
                &raw_scalars,
                TOL_4C_ABS,
                TOL_4C_REL,
                &format!("raw 4c {representation:?} shell interpretation shls={shls:?}"),
            );
        }
    }
}

#[test]
fn four_c_raw_optimizer_mode_is_invariant_and_wrapper_aligned() {
    let (atm, bas, env) = stable_raw_layout();
    let operator = four_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let baseline_options =
            optimizer_options(&["four-c-optimizer-off", representation_label(representation)]);
        let optimized_options =
            optimizer_options(&["four-c-optimizer-on", representation_label(representation)]);

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
        .unwrap_or_else(|err| panic!("baseline raw query failed for {representation:?}: {err:?}"));

        let optimizer_mode = optimizer_mode_for_four_c(representation);
        let optimizer_handle =
            wrapper_cint_from_raw_layout(&atm, &bas, &env, representation).optimizer("int4c1e");
        let optimizer_ptr = NonNull::new(optimizer_handle.as_ptr() as *mut c_void);
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
                cache: match optimizer_mode {
                    RouteOptimizerMode::Supported => Some(optimizer_query_cache.as_slice()),
                    RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                        None
                    }
                },
                opt: match optimizer_mode {
                    RouteOptimizerMode::Supported => optimizer_ptr,
                    RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                        None
                    }
                },
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| panic!("optimized raw query failed for {representation:?}: {err:?}"));

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
            panic!("baseline raw evaluate failed for {representation:?}: {err:?}")
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
                cache: match optimizer_mode {
                    RouteOptimizerMode::Supported => Some(optimized_cache.as_mut_slice()),
                    RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                        None
                    }
                },
                opt: match optimizer_mode {
                    RouteOptimizerMode::Supported => optimizer_ptr,
                    RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                        None
                    }
                },
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| {
            panic!("optimized raw evaluate failed for {representation:?}: {err:?}")
        });

        let (wrapper_scalars, wrapper_dims) =
            wrapper_four_c(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
        assert_eq!(baseline_workspace.dims, wrapper_dims);
        match optimizer_mode {
            RouteOptimizerMode::Supported => {
                assert!(
                    optimizer_ptr.is_some(),
                    "4c route claims optimizer support but wrapper optimizer returned null for {representation:?}",
                );
                assert!(optimized_workspace.has_opt);
                assert!(optimized_workspace.has_cache);
            }
            RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                assert!(!optimized_workspace.has_opt);
                assert!(!optimized_workspace.has_cache);
            }
        }
        assert_within_tolerance(
            &baseline_output,
            &optimized_output,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("raw 4c {representation:?} optimizer on/off invariance"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &baseline_output,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("raw 4c {representation:?} optimizer-off vs wrapper"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &optimized_output,
            TOL_4C_ABS,
            TOL_4C_REL,
            &format!("raw 4c {representation:?} optimizer-on vs wrapper"),
        );
    }
}

#[test]
fn four_c_query_and_evaluate_surfaces_follow_shared_route_policy() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let operator = four_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-route-policy");
        let expected_route_id = route_id_for_representation(representation);
        let route_key = CpuRouteKey::new(
            IntegralFamily::FourCenterOneElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        );
        let request =
            ExecutionRequest::from_safe(operator, representation, REDUCED_SAFE_SHLS, &options);

        let safe_route = resolve_safe_route(&request)
            .unwrap_or_else(|err| panic!("safe resolve failed for {representation:?}: {err:?}"));
        let raw_route = resolve_raw_route(&request)
            .unwrap_or_else(|err| panic!("raw resolve failed for {representation:?}: {err:?}"));
        let capi_route = resolve_capi_route(&request)
            .unwrap_or_else(|err| panic!("capi resolve failed for {representation:?}: {err:?}"));

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
        .unwrap_or_else(|err| panic!("safe query failed for {representation:?}: {err:?}"));
        let safe_eval = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {representation:?}: {err:?}"));
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
        .unwrap_or_else(|err| panic!("raw query failed for {representation:?}: {err:?}"));
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
        .unwrap_or_else(|err| panic!("raw evaluate failed for {representation:?}: {err:?}"));

        assert_eq!(raw_query.dims, raw_result.dims);
        assert_eq!(safe_query.dims, raw_query.dims);
        assert_eq!(safe_route.key, route_key);
        assert_eq!(raw_route.key, route_key);
        assert_eq!(capi_route.key, route_key);
    }
}

#[test]
fn four_c_supported_routes_are_not_legacy_seeded_fallback() {
    let basis = stable_safe_basis();
    let operator = four_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "four-c-regression-no-filler");
        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {representation:?}: {err:?}"));
        let dims = safe_tensor.dims;
        let actual = match safe_tensor.output {
            cintx::EvaluationOutput::Real(values) => values,
            cintx::EvaluationOutput::Spinor(_) => panic!("4c routes must be real-valued"),
        };
        let legacy =
            legacy_seeded_real_scalars(entry_symbol_for_representation(representation), &dims);
        let max_diff = max_abs_diff(&legacy, &actual);
        assert!(
            max_diff > 1e-9,
            "4c {representation:?} output still matches legacy seeded filler path; max_abs_diff={max_diff}",
        );
    }
}

#[test]
fn four_c_policy_rejections_are_deterministic() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let operator = four_c_operator();

    let no_feature_options = phase2_cpu_options(&["four-c-policy-no-feature"]);
    let err_without_feature = safe::query_workspace(
        &basis,
        operator,
        Representation::Cartesian,
        REDUCED_SAFE_SHLS,
        &no_feature_options,
    )
    .expect_err("4c query without with-4c1e must be unsupported by policy");
    assert!(matches!(
        err_without_feature.error,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            ..
        }
    ));

    let spinor_options = parity_options(Representation::Spinor, "four-c-policy-spinor");
    let spinor_err = safe::query_workspace(
        &basis,
        operator,
        Representation::Spinor,
        REDUCED_SAFE_SHLS,
        &spinor_options,
    )
    .expect_err("4c spinor must be blocked by policy");
    assert!(matches!(
        spinor_err.error,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            ..
        }
    ));

    let high_l_basis = outside_envelope_safe_basis();
    let high_l_options = parity_options(Representation::Cartesian, "four-c-policy-high-l");
    let high_l_safe_err = safe::query_workspace(
        &high_l_basis,
        operator,
        Representation::Cartesian,
        REDUCED_SAFE_SHLS,
        &high_l_options,
    )
    .expect_err("max(l)>4 must be rejected outside Validated4C1E envelope");
    assert_eq!(
        high_l_safe_err.error,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: "route is outside the Validated4C1E policy envelope",
        }
    );

    let (high_l_atm, high_l_bas, high_l_env) = outside_envelope_raw_layout();
    let high_l_raw_err = raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Cartesian,
        RawQueryRequest {
            shls: REDUCED_RAW_SHLS,
            dims: None,
            atm: &high_l_atm,
            bas: &high_l_bas,
            env: &high_l_env,
            out: None,
            cache: None,
            opt: None,
        },
        &high_l_options,
    )
    .expect_err("raw max(l)>4 must be rejected outside Validated4C1E envelope");
    assert_eq!(
        high_l_raw_err.error,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: "route is outside the Validated4C1E policy envelope",
        }
    );

    for surface in [RouteSurface::Safe, RouteSurface::Raw, RouteSurface::CAbi] {
        let err = resolve_route(
            CpuRouteKey::new(
                IntegralFamily::FourCenterOneElectron,
                OperatorKind::ElectronRepulsion,
                Representation::Spinor,
            ),
            surface,
        )
        .expect_err("4c spinor route must remain unsupported through shared policy");
        assert!(matches!(
            err,
            LibcintRsError::UnsupportedApi {
                api: "cpu.route",
                ..
            }
        ));
    }

    let no_feature_request = ExecutionRequest::from_safe(
        operator,
        Representation::Cartesian,
        REDUCED_SAFE_SHLS,
        &no_feature_options,
    );
    let no_feature_route_err = resolve_safe_route(&no_feature_request)
        .expect_err("safe resolver should reject 4c without with-4c1e feature");
    assert!(matches!(
        no_feature_route_err,
        LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            ..
        }
    ));
    for &representation in SUPPORTED_REPRESENTATIONS {
        for surface in [RouteSurface::Safe, RouteSurface::Raw, RouteSurface::CAbi] {
            let err = resolve_route(
                CpuRouteKey::new(
                    IntegralFamily::FourCenterOneElectron,
                    OperatorKind::ElectronRepulsion,
                    representation,
                ),
                surface,
            )
            .expect_err(
                "key-only resolver must reject optional 4c routes without feature/envelope context",
            );
            assert_eq!(
                err,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    reason: "route is unsupported by shared route coverage policy",
                }
            );
        }
    }

    let _ = raw::query_workspace_compat_with_sentinels(
        operator,
        Representation::Cartesian,
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
        &parity_options(Representation::Cartesian, "four-c-policy-supported"),
    )
    .expect("supported 4c route should remain queryable inside Validated4C1E");
}

fn evaluate_raw_four_c(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
    options: &WorkspaceQueryOptions,
) -> (raw::RawCompatWorkspace, Vec<f64>) {
    let operator = four_c_operator();
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
    .unwrap_or_else(|err| panic!("raw query failed for 4c/{representation:?} {shls:?}: {err:?}"));

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
        panic!("raw evaluate failed for 4c/{representation:?} {shls:?}: {err:?}")
    });

    (workspace, output)
}

fn wrapper_four_c(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, representation);
    let shls_slice = shell_ranges_4(shls);

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            cint.integrate("int4c1e", None, shls_slice).into()
        }
        Representation::Spinor => {
            panic!("4c spinor is outside the validated envelope and must remain unsupported")
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

fn shell_ranges_4(shls: &[i32]) -> [[usize; 2]; 4] {
    [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
        [shls[2] as usize, shls[2] as usize + 1],
        [shls[3] as usize, shls[3] as usize + 1],
    ]
}

fn parity_options(representation: Representation, gate: &'static str) -> WorkspaceQueryOptions {
    let mut flags = Vec::with_capacity(3);
    flags.push("with-4c1e");
    flags.push(gate);
    flags.push(representation_label(representation));
    phase2_cpu_options(flags.as_slice())
}

fn optimizer_options(extra_flags: &[&'static str]) -> WorkspaceQueryOptions {
    let mut flags = Vec::with_capacity(2 + extra_flags.len());
    flags.push("with-4c1e");
    flags.push("phase3-optimizer-equivalence");
    flags.extend_from_slice(extra_flags);
    phase2_cpu_options(flags.as_slice())
}

fn optimizer_mode_for_four_c(representation: Representation) -> RouteOptimizerMode {
    route_manifest_entries()
        .iter()
        .find(|entry| {
            entry.key
                == CpuRouteKey::new(
                    IntegralFamily::FourCenterOneElectron,
                    OperatorKind::ElectronRepulsion,
                    representation,
                )
                && entry.status == RouteStatus::Implemented
        })
        .map(|entry| entry.optimizer_mode)
        .unwrap_or_else(|| panic!("missing implemented 4c route metadata for {representation:?}"))
}

fn representation_label(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "cart",
        Representation::Spherical => "sph",
        Representation::Spinor => "spinor",
    }
}

fn route_id_for_representation(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "int4c1e_eri.cart.cpu.direct.v1",
        Representation::Spherical => "int4c1e_eri.sph.cpu.direct.v1",
        Representation::Spinor => panic!("4c spinor route must remain unsupported"),
    }
}

fn entry_symbol_for_representation(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "int4c1e_cart",
        Representation::Spherical => "int4c1e_sph",
        Representation::Spinor => panic!("4c spinor route must remain unsupported"),
    }
}

fn four_c_operator() -> Operator {
    Operator::new(
        IntegralFamily::FourCenterOneElectron,
        OperatorKind::ElectronRepulsion,
    )
    .expect("4c electron-repulsion operator should be valid")
}

fn outside_envelope_safe_basis() -> cintx::BasisSet {
    let atom_a = cintx::Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = cintx::Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");

    let shell_h =
        cintx::Shell::new(0, 5, vec![6.0, 1.2], vec![0.7, 0.3]).expect("h shell is valid");
    let shell_p =
        cintx::Shell::new(0, 1, vec![4.0, 0.8], vec![0.6, 0.4]).expect("p shell is valid");
    let shell_d =
        cintx::Shell::new(1, 2, vec![3.0, 0.7], vec![0.5, 0.5]).expect("d shell is valid");
    let shell_s =
        cintx::Shell::new(1, 0, vec![2.0, 0.5], vec![0.55, 0.45]).expect("s shell is valid");

    cintx::BasisSet::new(
        vec![atom_a, atom_b],
        vec![shell_h, shell_p, shell_d, shell_s],
    )
    .expect("outside-envelope fixture basis should be valid")
}

fn outside_envelope_raw_layout() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, mut bas, env) = stable_raw_layout();
    // First shell angular momentum becomes l=5, outside the Validated4C1E envelope.
    bas[1] = 5;
    (atm, bas, env)
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

fn col_major_to_row_major_real_rank4(values: &[f64], dims: &[usize]) -> Vec<f64> {
    let [d0, d1, d2, d3]: [usize; 4] = dims
        .try_into()
        .unwrap_or_else(|_| panic!("4c layouts must be rank-4, got {}", dims.len()));

    let mut row_major = vec![0.0f64; values.len()];
    for l in 0..d3 {
        for k in 0..d2 {
            for j in 0..d1 {
                for i in 0..d0 {
                    let col_index = i + d0 * (j + d1 * (k + d2 * l));
                    let row_index = ((i * d1 + j) * d2 + k) * d3 + l;
                    row_major[row_index] = values[col_index];
                }
            }
        }
    }

    row_major
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
