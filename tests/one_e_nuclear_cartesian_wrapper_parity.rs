#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{
    EvaluationOutput, EvaluationOutputMut, IntegralFamily, Operator, OperatorKind,
    RawEvaluateRequest, RawQueryRequest, Representation, WorkspaceQueryOptions, raw, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{
    phase2_cpu_options, phase3_optimizer_options, stable_raw_layout, stable_safe_basis,
};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

const REDUCED_D_S_SAFE_SHLS: &[usize] = &[2, 3];
const REDUCED_D_S_RAW_SHLS: &[i32] = &[2, 3];
const LAYOUT_D_P_RAW_SHLS: &[i32] = &[2, 1];

#[test]
fn safe_api_matches_wrapper_for_reduced_one_e_nuclear_cartesian_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-nuclear-cartesian-safe-wrapper"]);

    let safe_tensor = safe::evaluate(
        &basis,
        nuclear_operator(),
        Representation::Cartesian,
        REDUCED_D_S_SAFE_SHLS,
        &options,
    )
    .expect("safe evaluate must succeed for reduced Cartesian nuclear fixture");
    let safe_scalars = flatten_real_output(safe_tensor.output);

    let (wrapper_scalars, wrapper_dims) =
        wrapper_nuclear_cartesian(&atm, &bas, &env, REDUCED_D_S_RAW_SHLS);

    assert_eq!(safe_tensor.dims, vec![6, 1]);
    assert_eq!(safe_tensor.dims, wrapper_dims);
    assert_eq!(safe_scalars.len(), wrapper_scalars.len());

    assert_within_tolerance(
        &wrapper_scalars,
        &safe_scalars,
        "safe Cartesian nuclear vs wrapper reduced d/s case",
    );
}

#[test]
fn safe_evaluate_into_matches_wrapper_for_reduced_one_e_nuclear_cartesian_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-nuclear-cartesian-safe-evaluate-into-wrapper"]);
    let mut safe_output = vec![0.0f64; 6];

    let safe_meta = safe::evaluate_into(
        &basis,
        nuclear_operator(),
        Representation::Cartesian,
        REDUCED_D_S_SAFE_SHLS,
        &options,
        EvaluationOutputMut::Real(&mut safe_output),
    )
    .expect("safe evaluate_into must succeed for reduced Cartesian nuclear fixture");

    let (wrapper_scalars, wrapper_dims) =
        wrapper_nuclear_cartesian(&atm, &bas, &env, REDUCED_D_S_RAW_SHLS);

    assert_eq!(safe_meta.dims, vec![6, 1]);
    assert_eq!(safe_meta.dims, wrapper_dims);
    assert_eq!(safe_output.len(), wrapper_scalars.len());

    assert_within_tolerance(
        &wrapper_scalars,
        &safe_output,
        "safe evaluate_into Cartesian nuclear vs wrapper reduced d/s case",
    );
}

#[test]
fn raw_api_matches_wrapper_for_reduced_one_e_nuclear_cartesian_case() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-nuclear-cartesian-raw-wrapper"]);

    let (workspace, raw_scalars) =
        evaluate_raw_cartesian_nuclear(&atm, &bas, &env, REDUCED_D_S_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_nuclear_cartesian(&atm, &bas, &env, REDUCED_D_S_RAW_SHLS);

    assert_eq!(workspace.dims, vec![6, 1]);
    assert_eq!(workspace.dims, wrapper_dims);
    assert_eq!(raw_scalars.len(), wrapper_scalars.len());

    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw Cartesian nuclear vs wrapper reduced d/s case",
    );
}

#[test]
fn raw_api_uses_libcint_flat_layout_for_cartesian_nuclear_output() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-nuclear-cartesian-layout"]);

    let (workspace, raw_scalars) =
        evaluate_raw_cartesian_nuclear(&atm, &bas, &env, LAYOUT_D_P_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_nuclear_cartesian(&atm, &bas, &env, LAYOUT_D_P_RAW_SHLS);
    let row_major_scalars = col_major_to_row_major(&wrapper_scalars, &wrapper_dims);

    assert_eq!(workspace.dims, vec![6, 3]);
    assert_eq!(workspace.dims, wrapper_dims);
    assert_ne!(
        wrapper_scalars, row_major_scalars,
        "d/p fixture must distinguish libcint column-major from row-major flattening"
    );

    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw Cartesian nuclear must match libcint flat column-major layout",
    );

    let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
    assert!(
        row_major_max_diff > 1e-9,
        "raw output unexpectedly aligns with row-major flattening; max_abs_diff={row_major_max_diff}"
    );
}

#[test]
fn raw_cartesian_nuclear_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-nuclear-cartesian-raw-inputs"]);

    for &shls in &[
        &[0, 1][..],
        REDUCED_D_S_RAW_SHLS,
        LAYOUT_D_P_RAW_SHLS,
        &[3, 2][..],
    ] {
        let (workspace, raw_scalars) =
            evaluate_raw_cartesian_nuclear(&atm, &bas, &env, shls, &options);
        let (wrapper_scalars, wrapper_dims) = wrapper_nuclear_cartesian(&atm, &bas, &env, shls);

        assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw Cartesian nuclear shell interpretation shls={shls:?}"),
        );
    }
}

#[test]
fn raw_cartesian_nuclear_is_optimizer_invariant_against_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let baseline_options = phase3_optimizer_options(&["one-e-nuclear-cartesian-optimizer-off"]);
    let optimized_options = phase3_optimizer_options(&["one-e-nuclear-cartesian-optimizer-on"]);

    let baseline_workspace = raw::query_workspace_compat_with_sentinels(
        nuclear_operator(),
        Representation::Cartesian,
        RawQueryRequest {
            shls: REDUCED_D_S_RAW_SHLS,
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
    .expect("baseline raw query must succeed for reduced Cartesian nuclear fixture");
    let optimized_workspace = raw::query_workspace_compat_with_sentinels(
        nuclear_operator(),
        Representation::Cartesian,
        RawQueryRequest {
            shls: REDUCED_D_S_RAW_SHLS,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: None,
            cache: None,
            opt: None,
        },
        &optimized_options,
    )
    .expect("optimized raw query must succeed for reduced Cartesian nuclear fixture");

    let mut baseline_output = vec![0.0f64; baseline_workspace.required_bytes / 8];
    let mut optimized_output = vec![0.0f64; optimized_workspace.required_bytes / 8];
    raw::evaluate_compat(
        nuclear_operator(),
        Representation::Cartesian,
        &baseline_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_S_RAW_SHLS,
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
    .expect("baseline raw evaluate must succeed for reduced Cartesian nuclear fixture");
    raw::evaluate_compat(
        nuclear_operator(),
        Representation::Cartesian,
        &optimized_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_S_RAW_SHLS,
            dims: None,
            atm: &atm,
            bas: &bas,
            env: &env,
            out: &mut optimized_output,
            cache: None,
            opt: None,
        },
        &optimized_options,
    )
    .expect("optimized raw evaluate must succeed for reduced Cartesian nuclear fixture");

    let (wrapper_scalars, wrapper_dims) =
        wrapper_nuclear_cartesian(&atm, &bas, &env, REDUCED_D_S_RAW_SHLS);

    assert_eq!(baseline_workspace.dims, vec![6, 1]);
    assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
    assert_eq!(baseline_workspace.dims, wrapper_dims);
    assert!(!optimized_workspace.has_opt);
    assert!(!optimized_workspace.has_cache);
    assert_within_tolerance(
        &baseline_output,
        &optimized_output,
        "raw Cartesian nuclear optimizer on/off parity for reduced d/s case",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &baseline_output,
        "raw Cartesian nuclear baseline optimizer-off vs wrapper reduced d/s case",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &optimized_output,
        "raw Cartesian nuclear optimizer-on vs wrapper reduced d/s case",
    );
}

fn nuclear_operator() -> Operator {
    Operator::new(IntegralFamily::OneElectron, OperatorKind::NuclearAttraction)
        .expect("one-electron nuclear operator must be valid")
}

fn parity_options(extra_feature_flags: &[&'static str]) -> WorkspaceQueryOptions {
    phase2_cpu_options(extra_feature_flags)
}

fn evaluate_raw_cartesian_nuclear(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    options: &WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let workspace = raw::query_workspace_compat_with_sentinels(
        nuclear_operator(),
        Representation::Cartesian,
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
    .unwrap_or_else(|err| panic!("raw query failed for Cartesian nuclear {shls:?}: {err:?}"));

    let mut output = vec![0.0f64; workspace.required_bytes / core::mem::size_of::<f64>()];
    raw::evaluate_compat(
        nuclear_operator(),
        Representation::Cartesian,
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
    .unwrap_or_else(|err| panic!("raw evaluate failed for Cartesian nuclear {shls:?}: {err:?}"));

    (workspace, output)
}

fn wrapper_nuclear_cartesian(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env);
    let shls_slice = [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
    ];
    let (scalars, dims) = cint.integrate("int1e_nuc", None, shls_slice).into();
    (scalars, dims)
}

fn wrapper_cint_from_raw_layout(atm: &[i32], bas: &[i32], env: &[f64]) -> CInt {
    let atm_rows = atm
        .chunks_exact(6)
        .map(|row| row.try_into().expect("atm rows must have 6 slots"))
        .collect();
    let bas_rows = bas
        .chunks_exact(8)
        .map(|row| row.try_into().expect("bas rows must have 8 slots"))
        .collect();

    CInt {
        atm: atm_rows,
        bas: bas_rows,
        ecpbas: Vec::new(),
        env: env.to_vec(),
        cint_type: CIntType::Cartesian,
    }
}

fn flatten_real_output(output: EvaluationOutput) -> Vec<f64> {
    match output {
        EvaluationOutput::Real(values) => values,
        EvaluationOutput::Spinor(values) => {
            panic!(
                "Cartesian nuclear must produce real output, got spinor len={}",
                values.len()
            )
        }
    }
}

fn col_major_to_row_major(column_major: &[f64], dims: &[usize]) -> Vec<f64> {
    assert_eq!(dims.len(), 2, "1e nuclear fixture must be rank-2");
    let di = dims[0];
    let dj = dims[1];
    let mut row_major = vec![0.0f64; column_major.len()];
    for j in 0..dj {
        for i in 0..di {
            let col_major_index = i + di * j;
            let row_major_index = j + dj * i;
            row_major[row_major_index] = column_major[col_major_index];
        }
    }
    row_major
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f64, f64::max)
}

fn assert_within_tolerance(expected: &[f64], actual: &[f64], context: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: scalar length mismatch"
    );

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
        let abs_diff = (expected_value - actual_value).abs();
        if abs_diff <= ABS_TOLERANCE {
            continue;
        }

        let relative_scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        let rel_diff = abs_diff / relative_scale;
        assert!(
            rel_diff <= REL_TOLERANCE,
            "{context}: mismatch at index {index}: expected={expected_value}, got={actual_value}, abs_diff={abs_diff}, rel_diff={rel_diff}"
        );
    }
}
