#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{raw, safe};
use phase2_fixtures::{
    flatten_safe_output, phase2_cpu_options, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

#[test]
fn stable_family_safe_raw_numeric_and_layout_equivalence() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-safe-raw-equivalence"]);

    for row in stable_phase2_matrix() {
        let operator = row.operator();
        let row_id = row.id();

        let safe_eval = safe::evaluate(
            &basis,
            operator,
            row.representation,
            row.safe_shell_tuple,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {row_id}: {err:?}"));
        let safe_scalars = flatten_safe_output(safe_eval.output);

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
        let mut raw_output = vec![0.0f64; raw_query.required_bytes / 8];

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
            safe_eval.dims, raw_result.dims,
            "safe and raw dims must match for {row_id}"
        );
        assert_eq!(
            safe_scalars.len(),
            raw_output.len(),
            "safe and raw scalar lengths must match for {row_id}"
        );

        for (index, (&safe_value, &raw_value)) in
            safe_scalars.iter().zip(raw_output.iter()).enumerate()
        {
            assert_close(safe_value, raw_value, &row_id, index);
        }
    }
}

#[test]
fn safe_evaluate_into_layout_matches_raw_workspace_contract() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["phase2-safe-raw-layout-contract"]);

    for row in stable_phase2_matrix() {
        let operator = row.operator();
        let row_id = row.id();
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

        match row.representation {
            cintx::Representation::Cartesian | cintx::Representation::Spherical => {
                let mut output = vec![0.0; raw_query.required_elements];
                let metadata = safe::evaluate_into(
                    &basis,
                    operator,
                    row.representation,
                    row.safe_shell_tuple,
                    &options,
                    cintx::EvaluationOutputMut::Real(&mut output),
                )
                .unwrap_or_else(|err| panic!("safe evaluate_into failed for {row_id}: {err:?}"));

                assert_eq!(metadata.dims, raw_query.dims);
                assert_eq!(metadata.required_bytes, raw_query.required_bytes);
                assert_eq!(metadata.element_count, raw_query.required_elements);
            }
            cintx::Representation::Spinor => {
                let mut output = vec![[0.0, 0.0]; raw_query.required_elements];
                let metadata = safe::evaluate_into(
                    &basis,
                    operator,
                    row.representation,
                    row.safe_shell_tuple,
                    &options,
                    cintx::EvaluationOutputMut::Spinor(&mut output),
                )
                .unwrap_or_else(|err| panic!("safe evaluate_into failed for {row_id}: {err:?}"));

                assert_eq!(metadata.dims, raw_query.dims);
                assert_eq!(metadata.required_bytes, raw_query.required_bytes);
                assert_eq!(metadata.element_count, raw_query.required_elements);
            }
        }
    }
}

fn assert_close(expected: f64, got: f64, row_id: &str, index: usize) {
    let diff = (expected - got).abs();
    if diff <= ABS_TOLERANCE {
        return;
    }

    let scale = expected.abs().max(got.abs()).max(1.0);
    let relative = diff / scale;
    assert!(
        relative <= REL_TOLERANCE,
        "safe/raw mismatch for {row_id} at index {index}: expected={expected}, got={got}, abs_diff={diff}, rel_diff={relative}"
    );
}
