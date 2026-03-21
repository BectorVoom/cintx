#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{raw, safe};
use oracle_runner::{
    TolerancePolicy, assert_within_tolerance, oracle_expected_scalars_with_wrapper_override,
};
use phase2_fixtures::{
    flatten_safe_output, phase2_cpu_options, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};

#[test]
fn oracle_tolerance_matrix() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-oracle-tolerance"]);
    let tolerance = TolerancePolicy::strict();

    for row in stable_phase2_matrix() {
        let operator = row.operator();
        let row_id = row.id();

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

        let mut raw_scalars = vec![0.0f64; raw_workspace.required_bytes / 8];
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
                out: &mut raw_scalars,
                cache: None,
                opt: None,
            },
            &options,
        )
        .unwrap_or_else(|err| panic!("raw evaluate failed for {row_id}: {err:?}"));

        let oracle_scalars = oracle_expected_scalars_with_wrapper_override(
            row.route_key(),
            row.representation,
            &raw_workspace.dims,
        )
        .unwrap_or_else(|err| panic!("oracle generation failed for {row_id}: {err:?}"));

        assert_eq!(safe_tensor.dims, raw_workspace.dims);
        assert_eq!(safe_tensor.dims, raw_result.dims);
        assert_eq!(
            safe_scalars.len(),
            oracle_scalars.len(),
            "safe output length must match oracle length for {row_id}"
        );
        assert_eq!(
            raw_scalars.len(),
            oracle_scalars.len(),
            "raw output length must match oracle length for {row_id}"
        );

        assert_within_tolerance(
            &oracle_scalars,
            &safe_scalars,
            tolerance,
            &format!("{row_id} safe-vs-oracle"),
        );
        assert_within_tolerance(
            &oracle_scalars,
            &raw_scalars,
            tolerance,
            &format!("{row_id} raw-vs-oracle"),
        );
    }
}
