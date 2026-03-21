#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{IntegralFamily, LibcintRsError, OperatorKind, QueryDiagnostics, raw, safe};
use phase2_fixtures::{
    phase2_cpu_options, phase2_cpu_options_with_limit, representation_width_bytes,
    stable_phase2_matrix, stable_raw_layout, stable_safe_basis,
};

#[test]
fn safe_raw_query_memory_contract_parity_matrix() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-memory-contract-parity"]);

    for row in stable_phase2_matrix() {
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

        assert_eq!(safe_query.dims, raw_query.dims);
        assert_eq!(safe_query.element_count, raw_query.required_elements);
        assert_eq!(safe_query.required_bytes, raw_query.memory_required_bytes);
        assert_eq!(safe_query.scratch_bytes, raw_query.memory_scratch_bytes);
        assert_eq!(
            raw_query.required_bytes,
            safe_query.element_count * representation_width_bytes(row.representation)
        );
        assert!(
            raw_query.memory_working_set_bytes <= raw_query.memory_required_bytes,
            "working set should never exceed required bytes for {row_id}"
        );
        assert!(raw_query.chunk_elements > 0);
        assert!(raw_query.chunk_count > 0);
    }
}

#[test]
fn raw_query_memory_limit_failure_has_complete_diagnostics() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options_with_limit(1, &["phase2-memory-limit-query"]);

    let failure = raw::query_workspace_compat_with_sentinels(
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
    .expect_err("raw query should fail when memory limit is infeasible");

    assert!(matches!(
        failure.error,
        LibcintRsError::MemoryLimitExceeded { limit_bytes: 1, .. }
    ));
    assert_diagnostics_common(
        &failure.diagnostics,
        "raw.compat.query_workspace",
        "phase2-memory-limit-query",
    );
    assert_eq!(failure.diagnostics.memory_limit_bytes, Some(1));
    assert!(failure.diagnostics.required_bytes.is_some());
    assert!(!failure.diagnostics.dims.is_empty());
}

#[test]
fn raw_execute_memory_limit_failure_preserves_output_and_diagnostics() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let (atm, bas, env) = stable_raw_layout();
    let query_options = phase2_cpu_options(&["phase2-memory-limit-execute"]);
    let execute_options = phase2_cpu_options_with_limit(1, &["phase2-memory-limit-execute"]);

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
        &query_options,
    )
    .expect("raw query should succeed before execute-time memory failure");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![42.0f64; required_scalars];
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
            out: &mut output,
            cache: None,
            opt: None,
        },
        &execute_options,
    )
    .expect_err("execute should fail when execute-time memory policy becomes infeasible");

    assert!(matches!(
        failure.error,
        LibcintRsError::MemoryLimitExceeded { limit_bytes: 1, .. }
    ));
    assert!(
        output
            .iter()
            .all(|value| (*value - 42.0).abs() < f64::EPSILON),
        "memory-limit failure must not partially mutate output buffer"
    );
    assert_diagnostics_common(
        &failure.diagnostics,
        "raw.compat.evaluate",
        "phase2-memory-limit-execute",
    );
    assert_eq!(failure.diagnostics.memory_limit_bytes, Some(1));
    assert!(failure.diagnostics.required_bytes.is_some());
    assert_eq!(
        failure.diagnostics.provided_bytes,
        Some(required_scalars * 8)
    );
}

#[test]
fn allocation_failure_simulation_is_typed_for_safe_and_raw() {
    let case = one_electron_spherical_case();
    let operator = case.operator();
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options =
        phase2_cpu_options(&["phase2-allocation-contract", "simulate-allocation-failure"]);

    let safe_failure = safe::evaluate(
        &basis,
        operator,
        case.representation,
        case.safe_shell_tuple,
        &options,
    )
    .expect_err("safe evaluate should surface simulated allocation failure");
    assert!(matches!(
        safe_failure.error,
        LibcintRsError::AllocationFailure {
            operation: "safe.evaluate",
            ..
        }
    ));
    assert_diagnostics_common(
        &safe_failure.diagnostics,
        "safe.evaluate",
        "simulate-allocation-failure",
    );

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
    .expect("raw query should still succeed before simulated allocation failure");

    let required_scalars = queried.required_bytes / 8;
    let mut output = vec![99.0f64; required_scalars];
    let raw_failure = raw::evaluate_compat(
        operator,
        case.representation,
        &queried,
        raw::RawEvaluateRequest {
            shls: case.raw_shls,
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
    .expect_err("raw evaluate should surface simulated allocation failure");
    assert!(matches!(
        raw_failure.error,
        LibcintRsError::AllocationFailure {
            operation: "raw.compat.evaluate",
            ..
        }
    ));
    assert!(
        output
            .iter()
            .all(|value| (*value - 99.0).abs() < f64::EPSILON),
        "allocation failure must not partially mutate output buffer"
    );
    assert_diagnostics_common(
        &raw_failure.diagnostics,
        "raw.compat.evaluate",
        "simulate-allocation-failure",
    );
    assert_eq!(
        raw_failure.diagnostics.provided_bytes,
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
