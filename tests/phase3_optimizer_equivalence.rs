#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{raw, safe, Representation};
use oracle_runner::{assert_within_tolerance, oracle_expected_scalars, TolerancePolicy};
use phase2_fixtures::{
    flatten_safe_output, phase3_optimizer_options, raw_optimizer_cache_len, stable_phase2_matrix,
    stable_raw_layout, stable_safe_basis,
};

#[test]
fn optimizer_on_off_equivalence_matrix() {
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

        let optimizer_query_cache = vec![0.0f64; raw_optimizer_cache_len(row.raw_shls)];
        let optimized_workspace = raw::query_workspace_compat_with_sentinels(
            operator,
            row.representation,
            raw::RawQueryRequest {
                shls: row.raw_shls,
                dims: None,
                atm: &atm,
                bas: &bas,
                env: &env,
                out: None,
                cache: Some(&optimizer_query_cache),
                opt: Some(NonNull::<c_void>::dangling()),
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| panic!("raw optimized query failed for {row_id}: {err:?}"));

        assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
        assert_eq!(
            baseline_workspace.required_bytes, optimized_workspace.required_bytes,
            "optimizer toggles must not change required bytes for {row_id}"
        );
        assert_eq!(
            optimized_workspace.cache_required_len,
            raw_optimizer_cache_len(row.raw_shls),
            "optimizer cache contract must track shell arity for {row_id}"
        );
        assert!(optimized_workspace.has_opt);
        assert!(optimized_workspace.has_cache);

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

        let optimized_result = raw::evaluate_compat(
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
                cache: Some(optimized_cache.as_mut_slice()),
                opt: Some(NonNull::<c_void>::dangling()),
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| panic!("raw optimized evaluate failed for {row_id}: {err:?}"));

        assert_eq!(safe_tensor.dims, baseline_workspace.dims);
        assert_eq!(safe_tensor.dims, baseline_result.dims);
        assert_eq!(safe_tensor.dims, optimized_result.dims);
        assert_eq!(safe_scalars.len(), baseline_output.len());
        assert_eq!(baseline_output.len(), optimized_output.len());
        let oracle_scalars =
            oracle_expected_scalars(row.route_key(), row.representation, &safe_tensor.dims)
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
