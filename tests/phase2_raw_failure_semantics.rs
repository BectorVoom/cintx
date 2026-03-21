#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{IntegralFamily, LibcintRsError, OperatorKind, QueryDiagnostics, raw};
use phase2_fixtures::{phase2_cpu_options, stable_phase2_matrix, stable_raw_layout};

#[test]
fn raw_execute_undersized_output_reports_typed_failure_and_preserves_buffer() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-raw-failure-undersized"]);

    let queried = raw::query_workspace_compat_with_sentinels(
        operator,
        case.representation,
        raw::RawQueryRequest {
            shls: case.raw_shls,
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
    .expect("raw query should succeed before undersized-output failure test");

    let required_scalars = queried.required_bytes / 8;
    let mut undersized = vec![13.5f64; required_scalars - 1];
    let failure = raw::evaluate_compat(
        operator,
        case.representation,
        &queried,
        raw::RawEvaluateRequest {
            shls: case.raw_shls,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut undersized,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("raw execute must fail for undersized output");

    assert!(matches!(
        failure.error,
        LibcintRsError::InvalidLayout {
            item: "out_length",
            expected,
            got
        } if expected == required_scalars && got == (required_scalars - 1)
    ));
    assert!(
        undersized
            .iter()
            .all(|value| (*value - 13.5).abs() < f64::EPSILON),
        "undersized failure must not partially write user output"
    );
    assert_diagnostics_common(
        &failure.diagnostics,
        "raw.compat.evaluate",
        "phase2-raw-failure-undersized",
    );
    assert_eq!(failure.diagnostics.dims, queried.dims);
    assert_eq!(
        failure.diagnostics.required_bytes,
        Some(queried.required_bytes)
    );
    assert_eq!(
        failure.diagnostics.provided_bytes,
        Some((required_scalars - 1) * 8)
    );
}

#[test]
fn raw_execute_dims_mismatch_is_typed_and_preserves_buffer() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-raw-failure-dims"]);

    let queried = raw::query_workspace_compat_with_sentinels(
        operator,
        case.representation,
        raw::RawQueryRequest {
            shls: case.raw_shls,
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
    .expect("raw query should succeed before dims mismatch test");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![7.0f64; required_scalars];
    let mut bad_dims = queried
        .dims
        .iter()
        .map(|dim| i32::try_from(*dim).unwrap_or(i32::MAX))
        .collect::<Vec<_>>();
    bad_dims[0] += 1;

    let failure = raw::evaluate_compat(
        operator,
        case.representation,
        &queried,
        raw::RawEvaluateRequest {
            shls: case.raw_shls,
            dims: Some(bad_dims.as_slice()),
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut output,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("raw execute must fail for dims mismatch");

    assert!(matches!(
        failure.error,
        LibcintRsError::DimsBufferMismatch { .. }
    ));
    assert!(
        output
            .iter()
            .all(|value| (*value - 7.0).abs() < f64::EPSILON),
        "dims mismatch must not partially write user output"
    );
    assert_diagnostics_common(
        &failure.diagnostics,
        "raw.compat.evaluate",
        "phase2-raw-failure-dims",
    );
    assert_eq!(
        failure.diagnostics.dims,
        bad_dims
            .iter()
            .map(|dim| usize::try_from(*dim).unwrap_or(usize::MAX))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        failure.diagnostics.provided_bytes,
        Some(required_scalars * 8),
        "provided output bytes should still be reported on dims mismatch"
    );
}

#[test]
fn raw_query_execute_contract_violation_reports_typed_shell_tuple_error() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-raw-failure-contract"]);

    let queried = raw::query_workspace_compat_with_sentinels(
        operator,
        case.representation,
        raw::RawQueryRequest {
            shls: case.raw_shls,
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
    .expect("raw query should succeed before contract mismatch test");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![5.0f64; required_scalars];
    let swapped_shls = [case.raw_shls[1], case.raw_shls[0]];
    let failure = raw::evaluate_compat(
        operator,
        case.representation,
        &queried,
        raw::RawEvaluateRequest {
            shls: &swapped_shls,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut output,
            cache: None,
            opt: None,
        },
        &options,
    )
    .expect_err("raw execute must fail when execute shell tuple diverges from query");

    assert!(matches!(
        failure.error,
        LibcintRsError::InvalidInput { field: "shls", .. }
    ));
    assert!(
        output
            .iter()
            .all(|value| (*value - 5.0).abs() < f64::EPSILON),
        "query/execute contract violations must not partially write output"
    );
    assert_diagnostics_common(
        &failure.diagnostics,
        "raw.compat.evaluate",
        "phase2-raw-failure-contract",
    );
    assert_eq!(failure.diagnostics.shell_tuple, vec![1, 0]);
    assert_eq!(
        failure.diagnostics.required_bytes,
        Some(queried.required_bytes)
    );
    assert_eq!(
        failure.diagnostics.provided_bytes,
        Some(required_scalars * 8)
    );
}

fn one_electron_spherical_case() -> phase2_fixtures::StableMatrixCase {
    stable_phase2_matrix()
        .into_iter()
        .find(|case| {
            case.family == IntegralFamily::OneElectron
                && case.operator_kind == OperatorKind::Overlap
                && case.representation == cintx::Representation::Spherical
        })
        .expect("fixture matrix should include one-electron spherical case")
}

fn assert_diagnostics_common(diagnostics: &QueryDiagnostics, api: &str, feature_flag: &str) {
    assert_eq!(diagnostics.api, api);
    assert!(!diagnostics.representation.is_empty());
    assert!(!diagnostics.shell_tuple.is_empty());
    assert!(diagnostics.correlation_id != 0);
    assert_eq!(diagnostics.backend_candidate, "cpu");
    assert!(
        diagnostics.feature_flags.contains(&feature_flag),
        "diagnostics must preserve feature flags for failure triage"
    );
}
