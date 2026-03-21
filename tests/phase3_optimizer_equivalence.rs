#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    CpuRouteKey, EvaluationOutputMut, IntegralFamily, LibcintRsError, Representation,
    RouteOptimizerMode, RouteStatus, raw, route_manifest_entries, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use oracle_runner::{
    PHASE3_REQUIRED_GATE_REQUIREMENTS, TolerancePolicy, assert_requirement_traceability,
    assert_within_tolerance, oracle_expected_scalars_with_wrapper_override,
    phase3_oracle_profile_matrix,
};
use phase2_fixtures::{
    flatten_safe_output, phase3_optimizer_options, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};

#[test]
fn optimizer_on_off_equivalence_matrix() {
    for profile_case in phase3_oracle_profile_matrix() {
        assert_requirement_traceability(
            profile_case.requirement_ids,
            PHASE3_REQUIRED_GATE_REQUIREMENTS,
            &format!(
                "optimizer preflight profile {}",
                profile_case.profile.as_str()
            ),
        );
    }

    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let baseline_options = phase3_optimizer_options(&["phase3-optimizer-off"]);
    let optimized_options = phase3_optimizer_options(&["phase3-optimizer-on"]);
    let tolerance = TolerancePolicy::strict();

    for row in stable_phase2_matrix() {
        let operator = row.operator();
        let row_id = row.id();

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            row.representation,
            row.safe_shell_tuple,
            &baseline_options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {row_id}: {err:?}"));
        let safe_scalars = flatten_safe_output(safe_tensor.output);

        let baseline_workspace = raw::query_workspace_compat_with_sentinels(
            operator,
            row.representation,
            raw::RawQueryRequest {
                shls: row.raw_shls,
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
        .unwrap_or_else(|err| panic!("raw baseline query failed for {row_id}: {err:?}"));

        let optimizer_mode = optimizer_mode_for_route(row.route_key());
        let optimizer_handle = match optimizer_mode {
            RouteOptimizerMode::Supported => Some(optimizer_handle_for_case(
                row.family,
                row.representation,
                &atm,
                &bas,
                &env,
            )),
            RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => None,
        };
        let optimizer_ptr = optimizer_handle
            .as_ref()
            .and_then(|handle| NonNull::new(handle.as_ptr() as *mut c_void));
        if optimizer_mode == RouteOptimizerMode::Supported {
            assert!(
                optimizer_ptr.is_some(),
                "route {row_id} claims optimizer support but wrapper optimizer returned null",
            );
        }
        let optimizer_query_cache = vec![0.0f64; baseline_workspace.cache_required_len];
        let optimized_workspace =
            raw::query_workspace_compat_with_sentinels(
                operator,
                row.representation,
                raw::RawQueryRequest {
                    shls: row.raw_shls,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: None,
                    cache: match optimizer_mode {
                        RouteOptimizerMode::Supported => Some(&optimizer_query_cache),
                        RouteOptimizerMode::IgnoredButInvariant
                        | RouteOptimizerMode::NotApplicable => None,
                    },
                    opt: match optimizer_mode {
                        RouteOptimizerMode::Supported => optimizer_ptr,
                        RouteOptimizerMode::IgnoredButInvariant
                        | RouteOptimizerMode::NotApplicable => None,
                    },
                },
                &optimized_options,
            )
            .unwrap_or_else(|err| panic!("raw optimized query failed for {row_id}: {err:?}"));

        assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
        assert_eq!(
            baseline_workspace.required_bytes, optimized_workspace.required_bytes,
            "optimizer toggles must not change required bytes for {row_id}"
        );
        match optimizer_mode {
            RouteOptimizerMode::Supported => {
                assert!(
                    optimized_workspace.cache_required_len >= baseline_workspace.cache_required_len,
                    "optimizer cache contract cannot shrink for {row_id}"
                );
                assert!(optimized_workspace.has_opt);
                assert!(optimized_workspace.has_cache);
            }
            RouteOptimizerMode::IgnoredButInvariant | RouteOptimizerMode::NotApplicable => {
                assert!(!optimized_workspace.has_opt);
                assert!(!optimized_workspace.has_cache);
            }
        }

        let required_scalars = baseline_workspace.required_bytes / 8;
        let mut baseline_output = vec![0.0f64; required_scalars];
        let mut optimized_output = vec![0.0f64; required_scalars];
        let mut optimized_cache = vec![0.0f64; optimized_workspace.cache_required_len];

        let baseline_result = raw::evaluate_compat(
            operator,
            row.representation,
            &baseline_workspace,
            raw::RawEvaluateRequest {
                shls: row.raw_shls,
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
        .unwrap_or_else(|err| panic!("raw baseline evaluate failed for {row_id}: {err:?}"));

        let optimized_result =
            raw::evaluate_compat(
                operator,
                row.representation,
                &optimized_workspace,
                raw::RawEvaluateRequest {
                    shls: row.raw_shls,
                    dims: None,
                    atm: &atm,
                    bas: &bas,
                    env: &env,
                    out: &mut optimized_output,
                    cache: match optimizer_mode {
                        RouteOptimizerMode::Supported => Some(optimized_cache.as_mut_slice()),
                        RouteOptimizerMode::IgnoredButInvariant
                        | RouteOptimizerMode::NotApplicable => None,
                    },
                    opt: match optimizer_mode {
                        RouteOptimizerMode::Supported => optimizer_ptr,
                        RouteOptimizerMode::IgnoredButInvariant
                        | RouteOptimizerMode::NotApplicable => None,
                    },
                },
                &optimized_options,
            )
            .unwrap_or_else(|err| panic!("raw optimized evaluate failed for {row_id}: {err:?}"));

        assert_eq!(safe_tensor.dims, baseline_workspace.dims);
        assert_eq!(safe_tensor.dims, baseline_result.dims);
        assert_eq!(safe_tensor.dims, optimized_result.dims);
        assert_eq!(safe_scalars.len(), baseline_output.len());
        assert_eq!(baseline_output.len(), optimized_output.len());
        let oracle_scalars = oracle_expected_scalars_with_wrapper_override(
            row.route_key(),
            row.representation,
            &safe_tensor.dims,
        )
        .unwrap_or_else(|err| panic!("oracle generation failed for {row_id}: {err:?}"));
        assert_eq!(oracle_scalars.len(), baseline_output.len());

        assert_within_tolerance(
            &baseline_output,
            &optimized_output,
            tolerance,
            &format!("RAW-04 optimizer on/off parity {row_id}"),
        );
        assert_within_tolerance(
            &safe_scalars,
            &baseline_output,
            tolerance,
            &format!("RAW-04 safe baseline parity {row_id}"),
        );
        assert_within_tolerance(
            &safe_scalars,
            &optimized_output,
            tolerance,
            &format!("RAW-04 safe optimized parity {row_id}"),
        );
        assert_within_tolerance(
            &oracle_scalars,
            &baseline_output,
            tolerance,
            &format!("RAW-04 oracle baseline parity {row_id}"),
        );
        assert_within_tolerance(
            &oracle_scalars,
            &optimized_output,
            tolerance,
            &format!("RAW-04 oracle optimized parity {row_id}"),
        );

        if row.representation == Representation::Spinor {
            assert_eq!(
                baseline_output.len() % 2,
                0,
                "spinor scalar layout must remain real/imag paired for {row_id}"
            );
            assert_eq!(
                optimized_output.len() % 2,
                0,
                "spinor scalar layout must remain real/imag paired for {row_id}"
            );
        }
    }
}

#[test]
fn spinor_layout_and_oom_semantics_regression() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let row = stable_phase2_matrix()
        .into_iter()
        .find(|case| {
            case.family == cintx::IntegralFamily::ThreeCenterTwoElectron
                && case.representation == Representation::Spinor
        })
        .expect("stable matrix must include explicit supported 3c2e spinor row");
    let operator = row.operator();
    let row_id = row.id();
    let options = phase3_optimizer_options(&["phase3-spinor-layout-regression"]);
    let tolerance = TolerancePolicy::strict();

    let safe_tensor = safe::evaluate(
        &basis,
        operator,
        row.representation,
        row.safe_shell_tuple,
        &options,
    )
    .unwrap_or_else(|err| panic!("safe evaluate failed for {row_id}: {err:?}"));
    let safe_scalars = flatten_safe_output(safe_tensor.output);

    let raw_workspace = raw::query_workspace_compat_with_sentinels(
        operator,
        row.representation,
        raw::RawQueryRequest {
            shls: row.raw_shls,
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
    .unwrap_or_else(|err| panic!("raw query failed for {row_id}: {err:?}"));

    assert_eq!(safe_tensor.dims, raw_workspace.dims);
    assert_eq!(
        raw_workspace.required_bytes,
        safe_scalars.len() * 8,
        "spinor scalar layout must remain stable for {row_id}"
    );
    assert_eq!(
        raw_workspace.required_elements * 2,
        safe_scalars.len(),
        "spinor scalar layout must remain real/imag paired for {row_id}"
    );
    assert!(
        safe_scalars.iter().any(|value| *value != 0.0),
        "3c2e spinor safe output must contain non-zero payload values",
    );

    let required_scalars = raw_workspace.required_bytes / 8;
    let mut raw_output = vec![0.0f64; required_scalars];
    let raw_result = raw::evaluate_compat(
        operator,
        row.representation,
        &raw_workspace,
        raw::RawEvaluateRequest {
            shls: row.raw_shls,
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
    .unwrap_or_else(|err| panic!("raw evaluate failed for {row_id}: {err:?}"));
    assert_eq!(raw_result.dims, raw_workspace.dims);
    assert_eq!(raw_output.len() % 2, 0);
    assert!(
        raw_output.iter().any(|value| *value != 0.0),
        "3c2e spinor raw output must contain non-zero payload values",
    );

    let oracle_scalars = oracle_expected_scalars_with_wrapper_override(
        row.route_key(),
        row.representation,
        &safe_tensor.dims,
    )
    .unwrap_or_else(|err| panic!("oracle generation failed for {row_id}: {err:?}"));
    assert_within_tolerance(
        &oracle_scalars,
        &safe_scalars,
        tolerance,
        &format!("VERI-03 safe spinor layout regression {row_id}"),
    );
    assert_within_tolerance(
        &oracle_scalars,
        &raw_output,
        tolerance,
        &format!("VERI-03 raw spinor layout regression {row_id}"),
    );

    let mut undersized_safe = vec![[17.0, -17.0]; raw_workspace.required_elements - 1];
    let safe_contract_failure = safe::evaluate_into(
        &basis,
        operator,
        row.representation,
        row.safe_shell_tuple,
        &options,
        EvaluationOutputMut::Spinor(&mut undersized_safe),
    )
    .expect_err("undersized safe spinor output must fail before writing");
    assert!(matches!(
        safe_contract_failure.error,
        LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected,
            got
        } if expected == raw_workspace.required_elements
            && got == (raw_workspace.required_elements - 1)
    ));
    assert!(
        undersized_safe
            .iter()
            .all(|value| (value[0] - 17.0).abs() < f64::EPSILON
                && (value[1] + 17.0).abs() < f64::EPSILON),
        "safe contract failures must preserve caller buffers without partial writes",
    );

    let mut undersized_raw = vec![13.0f64; required_scalars - 1];
    let raw_contract_failure = raw::evaluate_compat(
        operator,
        row.representation,
        &raw_workspace,
        raw::RawEvaluateRequest {
            shls: row.raw_shls,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut undersized_raw,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("undersized raw output must fail before writing");
    assert!(matches!(
        raw_contract_failure.error,
        LibcintRsError::InvalidLayout {
            item: "out_length",
            expected,
            got
        } if expected == required_scalars && got == (required_scalars - 1)
    ));
    assert!(
        undersized_raw
            .iter()
            .all(|value| (*value - 13.0).abs() < f64::EPSILON),
        "raw contract failures must preserve caller buffers without partial writes",
    );

    let oom_options = phase3_optimizer_options(&[
        "phase3-oom-semantics-regression",
        "simulate-allocation-failure",
    ]);
    let safe_oom = safe::evaluate(
        &basis,
        operator,
        row.representation,
        row.safe_shell_tuple,
        &oom_options,
    )
    .expect_err("safe OOM simulation must return typed allocation failure");
    assert!(matches!(
        safe_oom.error,
        LibcintRsError::AllocationFailure {
            operation: "safe.evaluate",
            ..
        }
    ));

    let raw_oom_workspace = raw::query_workspace_compat_with_sentinels(
        operator,
        row.representation,
        raw::RawQueryRequest {
            shls: row.raw_shls,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: None,
            cache: None,
            opt: None,
        },
        &oom_options,
    )
    .expect("raw query should still succeed before simulated execute OOM");
    let mut raw_oom_output = vec![29.0f64; raw_oom_workspace.required_bytes / 8];
    let raw_oom = raw::evaluate_compat(
        operator,
        row.representation,
        &raw_oom_workspace,
        raw::RawEvaluateRequest {
            shls: row.raw_shls,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut raw_oom_output,
            cache: None,
            opt: None,
        },
        &oom_options,
    )
    .expect_err("raw OOM simulation must return typed allocation failure");
    assert!(matches!(
        raw_oom.error,
        LibcintRsError::AllocationFailure {
            operation: "raw.compat.evaluate",
            ..
        }
    ));
    assert!(
        raw_oom_output
            .iter()
            .all(|value| (*value - 29.0).abs() < f64::EPSILON),
        "raw OOM failures must preserve caller buffers without partial writes",
    );
}

fn optimizer_handle_for_case(
    family: IntegralFamily,
    representation: Representation,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> libcint::cint::CIntOptimizer {
    wrapper_cint_from_raw_layout(atm, bas, env, representation)
        .optimizer(optimizer_integral(family))
}

fn optimizer_mode_for_route(route_key: CpuRouteKey) -> RouteOptimizerMode {
    route_manifest_entries()
        .iter()
        .find(|entry| entry.key == route_key && entry.status == RouteStatus::Implemented)
        .map(|entry| entry.optimizer_mode)
        .unwrap_or_else(|| {
            panic!(
                "missing implemented route metadata for {:?}/{:?}/{:?}",
                route_key.family, route_key.operator, route_key.representation
            )
        })
}

fn optimizer_integral(family: IntegralFamily) -> &'static str {
    match family {
        IntegralFamily::TwoElectron => "int2e",
        IntegralFamily::TwoCenterTwoElectron => "int2c2e",
        IntegralFamily::ThreeCenterOneElectron => "int3c1e_p2",
        IntegralFamily::ThreeCenterTwoElectron => "int3c2e_ip1",
        IntegralFamily::FourCenterOneElectron => "int4c1e",
        IntegralFamily::OneElectron => "int1e_ovlp",
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
