#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{ExecutionBackend, LibcintRsError, raw, route, safe};
use oracle_runner::{
    TolerancePolicy, assert_within_tolerance, oracle_expected_scalars_with_wrapper_override,
};
use phase2_fixtures::{
    flatten_safe_output, out_of_phase_route_keys, phase2_cpu_options, representation_width_bytes,
    stable_phase2_matrix, stable_raw_layout, stable_safe_basis,
};

#[test]
fn cpu_execution_matrix_stable_family_envelopes() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-cpu-execution-matrix"]);
    let matrix = stable_phase2_matrix();
    let tolerance = TolerancePolicy::strict();

    assert_eq!(
        matrix.len(),
        14,
        "stable-family matrix must include all currently supported envelopes"
    );

    for row in matrix {
        let operator = row.operator();
        let row_id = row.id();

        let safe_query = safe::query_workspace(
            &basis,
            operator,
            row.representation,
            row.safe_shell_tuple,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe query failed for {row_id}: {err:?}"));
        let safe_eval = safe::evaluate(
            &basis,
            operator,
            row.representation,
            row.safe_shell_tuple,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {row_id}: {err:?}"));
        let safe_scalars = flatten_safe_output(safe_eval.output);
        let expected_output_bytes =
            safe_query.element_count * representation_width_bytes(row.representation);

        assert_eq!(
            safe_eval.dims, safe_query.dims,
            "safe evaluate dims must match safe query for {row_id}"
        );
        assert_eq!(
            safe_scalars.len(),
            expected_output_bytes / 8,
            "safe output scalar count must match queried bytes for {row_id}"
        );
        let oracle_scalars = oracle_expected_scalars_with_wrapper_override(
            row.route_key(),
            row.representation,
            &safe_eval.dims,
        )
        .unwrap_or_else(|err| panic!("oracle generation failed for {row_id}: {err:?}"));
        assert_eq!(
            oracle_scalars.len(),
            safe_scalars.len(),
            "oracle scalar count must match safe output layout for {row_id}"
        );
        assert_within_tolerance(
            &oracle_scalars,
            &safe_scalars,
            tolerance,
            &format!("phase2 safe oracle parity {row_id}"),
        );

        let raw_query = raw::query_workspace_compat_with_sentinels(
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

        assert_eq!(
            raw_query.dims, safe_query.dims,
            "raw and safe dimensions must agree for {row_id}"
        );
        assert_eq!(
            raw_query.required_bytes, expected_output_bytes,
            "raw required bytes must match output layout contract for {row_id}"
        );
        assert_eq!(
            raw_query.required_elements, safe_query.element_count,
            "raw and safe element counts must agree for {row_id}"
        );

        let required_scalars = raw_query.required_bytes / 8;
        let mut raw_output = vec![0.0f64; required_scalars + 2];
        raw_output[required_scalars] = 111.0;
        raw_output[required_scalars + 1] = 222.0;

        let raw_result = raw::evaluate_compat(
            operator,
            row.representation,
            &raw_query,
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

        assert_eq!(
            raw_result.dispatch.backend,
            ExecutionBackend::CpuReference,
            "all phase-2 rows must execute through CPU baseline backend"
        );
        assert_eq!(raw_result.dims, safe_query.dims);
        assert_eq!(raw_result.required_elements, raw_query.required_elements);
        assert_eq!(raw_result.required_bytes, raw_query.required_bytes);
        assert_within_tolerance(
            &oracle_scalars,
            &raw_output[..required_scalars],
            tolerance,
            &format!("phase2 raw oracle parity {row_id}"),
        );
        assert_eq!(
            raw_output[required_scalars], 111.0,
            "raw execute must not overwrite bytes outside required output span for {row_id}"
        );
        assert_eq!(
            raw_output[required_scalars + 1],
            222.0,
            "raw execute must not overwrite bytes outside required output span for {row_id}"
        );
    }
}

#[test]
fn out_of_phase_envelopes_report_typed_unsupported() {
    for key in out_of_phase_route_keys() {
        let err = route(key).expect_err("out-of-phase route must fail with typed unsupported");
        assert!(
            matches!(
                err,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "expected typed UnsupportedApi for out-of-phase key {key:?}, got {err:?}"
        );
    }
}
