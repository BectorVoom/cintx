#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    EvaluationOutput, EvaluationOutputMut, IntegralFamily, Operator, OperatorKind,
    RawEvaluateRequest, RawQueryRequest, Representation, WorkspaceQueryOptions, raw, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{
    phase2_cpu_options, phase3_optimizer_options, raw_optimizer_cache_len, stable_raw_layout,
    stable_safe_basis,
};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

const REDUCED_D_P_SAFE_SHLS: &[usize] = &[2, 1];
const REDUCED_D_P_RAW_SHLS: &[i32] = &[2, 1];

#[test]
fn spherical_safe_evaluate_matches_wrapper_for_reduced_one_e_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spherical-safe-wrapper"]);

    let safe_tensor = safe::evaluate(
        &basis,
        kinetic_operator(),
        Representation::Spherical,
        REDUCED_D_P_SAFE_SHLS,
        &options,
    )
    .expect("safe spherical kinetic evaluate must succeed");
    let safe_scalars = flatten_real_output(safe_tensor.output);

    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(safe_tensor.dims, wrapper_dims);
    assert_eq!(safe_scalars.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &safe_scalars,
        "safe spherical kinetic vs wrapper reduced d/p case",
    );
}

#[test]
fn spherical_safe_evaluate_into_matches_wrapper_for_reduced_one_e_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spherical-safe-evaluate-into-wrapper"]);

    let (_, wrapper_dims) = wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);
    let element_count = checked_product(&wrapper_dims);
    let mut safe_output = vec![0.0f64; element_count];

    let safe_meta = safe::evaluate_into(
        &basis,
        kinetic_operator(),
        Representation::Spherical,
        REDUCED_D_P_SAFE_SHLS,
        &options,
        EvaluationOutputMut::Real(&mut safe_output),
    )
    .expect("safe spherical kinetic evaluate_into must succeed");

    let (wrapper_scalars, expected_dims) =
        wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(safe_meta.dims, expected_dims);
    assert_eq!(safe_output.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &safe_output,
        "safe spherical kinetic evaluate_into vs wrapper reduced d/p case",
    );
}

#[test]
fn spherical_raw_matches_wrapper_for_reduced_one_e_case() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spherical-raw-wrapper"]);

    let (workspace, raw_scalars) =
        evaluate_raw_spherical_kinetic(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(workspace.dims, wrapper_dims);
    assert_eq!(raw_scalars.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw spherical kinetic vs wrapper reduced d/p case",
    );
}

#[test]
fn spherical_raw_uses_libcint_flat_layout() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spherical-layout"]);

    let (workspace, raw_scalars) =
        evaluate_raw_spherical_kinetic(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);
    let row_major_scalars = col_major_to_row_major_real(&wrapper_scalars, &wrapper_dims);

    assert_eq!(workspace.dims, wrapper_dims);
    assert_ne!(
        wrapper_scalars, row_major_scalars,
        "d/p spherical fixture must distinguish column-major from row-major flattening"
    );

    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw spherical kinetic must preserve libcint flat column-major layout",
    );

    let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
    assert!(
        row_major_max_diff > 1e-9,
        "raw spherical kinetic unexpectedly aligns with row-major flattening"
    );
}

#[test]
fn spherical_raw_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spherical-raw-inputs"]);

    for &shls in &[&[0, 1][..], REDUCED_D_P_RAW_SHLS, &[2, 2][..], &[3, 2][..]] {
        let (workspace, raw_scalars) =
            evaluate_raw_spherical_kinetic(&atm, &bas, &env, shls, &options);
        let (wrapper_scalars, wrapper_dims) = wrapper_kinetic_spherical(&atm, &bas, &env, shls);

        assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw spherical kinetic shell interpretation shls={shls:?}"),
        );
    }
}

#[test]
fn spherical_raw_is_optimizer_invariant_against_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let baseline_options = phase3_optimizer_options(&["one-e-kinetic-spherical-optimizer-off"]);
    let optimized_options = phase3_optimizer_options(&["one-e-kinetic-spherical-optimizer-on"]);

    let baseline_workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spherical,
        RawQueryRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("baseline raw spherical query must succeed");

    let optimizer_query_cache = vec![0.0f64; raw_optimizer_cache_len(REDUCED_D_P_RAW_SHLS)];
    let optimized_workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spherical,
        RawQueryRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("optimized raw spherical query must succeed");

    let mut baseline_output = vec![0.0f64; baseline_workspace.required_bytes / 8];
    let mut optimized_output = vec![0.0f64; optimized_workspace.required_bytes / 8];
    let mut optimized_cache = vec![0.0f64; optimized_workspace.cache_required_len];

    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spherical,
        &baseline_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("baseline raw spherical evaluate must succeed");

    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spherical,
        &optimized_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("optimized raw spherical evaluate must succeed");

    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spherical(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
    assert_eq!(baseline_workspace.dims, wrapper_dims);
    assert_within_tolerance(
        &baseline_output,
        &optimized_output,
        "raw spherical kinetic optimizer on/off parity",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &baseline_output,
        "raw spherical kinetic optimizer-off vs wrapper",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &optimized_output,
        "raw spherical kinetic optimizer-on vs wrapper",
    );
}

#[test]
fn spinor_safe_evaluate_matches_wrapper_for_reduced_one_e_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spinor-safe-wrapper"]);

    let safe_tensor = safe::evaluate(
        &basis,
        kinetic_operator(),
        Representation::Spinor,
        REDUCED_D_P_SAFE_SHLS,
        &options,
    )
    .expect("safe spinor kinetic evaluate must succeed");
    let safe_scalars = flatten_spinor_output(safe_tensor.output);

    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(safe_tensor.dims, wrapper_dims);
    assert_eq!(safe_scalars.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &safe_scalars,
        "safe spinor kinetic vs wrapper reduced d/p case",
    );
}

#[test]
fn spinor_safe_evaluate_into_matches_wrapper_for_reduced_one_e_case() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spinor-safe-evaluate-into-wrapper"]);

    let (_, wrapper_dims) = wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);
    let element_count = checked_product(&wrapper_dims);
    let mut safe_output = vec![[0.0f64; 2]; element_count];

    let safe_meta = safe::evaluate_into(
        &basis,
        kinetic_operator(),
        Representation::Spinor,
        REDUCED_D_P_SAFE_SHLS,
        &options,
        EvaluationOutputMut::Spinor(&mut safe_output),
    )
    .expect("safe spinor kinetic evaluate_into must succeed");

    let (wrapper_scalars, expected_dims) =
        wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);
    let flattened_safe = flatten_spinor_pairs(&safe_output);

    assert_eq!(safe_meta.dims, expected_dims);
    assert_eq!(flattened_safe.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &flattened_safe,
        "safe spinor kinetic evaluate_into vs wrapper reduced d/p case",
    );
}

#[test]
fn spinor_raw_matches_wrapper_for_reduced_one_e_case() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spinor-raw-wrapper"]);

    let (workspace, raw_scalars) =
        evaluate_raw_spinor_kinetic(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(workspace.dims, wrapper_dims);
    assert_eq!(raw_scalars.len(), wrapper_scalars.len());
    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw spinor kinetic vs wrapper reduced d/p case",
    );
}

#[test]
fn spinor_raw_uses_libcint_flat_layout() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spinor-layout"]);

    let (workspace, raw_scalars) =
        evaluate_raw_spinor_kinetic(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS, &options);
    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);
    let row_major_scalars = col_major_to_row_major_spinor(&wrapper_scalars, &wrapper_dims);

    assert_eq!(workspace.dims, wrapper_dims);
    assert_ne!(
        wrapper_scalars, row_major_scalars,
        "d/p spinor fixture must distinguish column-major from row-major flattening"
    );

    assert_within_tolerance(
        &wrapper_scalars,
        &raw_scalars,
        "raw spinor kinetic must preserve libcint flat column-major layout",
    );

    let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
    assert!(
        row_major_max_diff > 1e-9,
        "raw spinor kinetic unexpectedly aligns with row-major flattening"
    );
}

#[test]
fn spinor_raw_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let options = parity_options(&["one-e-kinetic-spinor-raw-inputs"]);

    for &shls in &[&[0, 1][..], REDUCED_D_P_RAW_SHLS, &[2, 2][..], &[3, 2][..]] {
        let (workspace, raw_scalars) =
            evaluate_raw_spinor_kinetic(&atm, &bas, &env, shls, &options);
        let (wrapper_scalars, wrapper_dims) = wrapper_kinetic_spinor(&atm, &bas, &env, shls);

        assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw spinor kinetic shell interpretation shls={shls:?}"),
        );
    }
}

#[test]
fn spinor_raw_is_optimizer_invariant_against_wrapper() {
    let (atm, bas, env) = stable_raw_layout();
    let baseline_options = phase3_optimizer_options(&["one-e-kinetic-spinor-optimizer-off"]);
    let optimized_options = phase3_optimizer_options(&["one-e-kinetic-spinor-optimizer-on"]);

    let baseline_workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spinor,
        RawQueryRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("baseline raw spinor query must succeed");

    let optimizer_query_cache = vec![0.0f64; raw_optimizer_cache_len(REDUCED_D_P_RAW_SHLS)];
    let optimized_workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spinor,
        RawQueryRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("optimized raw spinor query must succeed");

    let mut baseline_output = vec![0.0f64; baseline_workspace.required_bytes / 8];
    let mut optimized_output = vec![0.0f64; optimized_workspace.required_bytes / 8];
    let mut optimized_cache = vec![0.0f64; optimized_workspace.cache_required_len];

    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spinor,
        &baseline_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("baseline raw spinor evaluate must succeed");

    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spinor,
        &optimized_workspace,
        RawEvaluateRequest {
            shls: REDUCED_D_P_RAW_SHLS,
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
    .expect("optimized raw spinor evaluate must succeed");

    let (wrapper_scalars, wrapper_dims) =
        wrapper_kinetic_spinor(&atm, &bas, &env, REDUCED_D_P_RAW_SHLS);

    assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
    assert_eq!(baseline_workspace.dims, wrapper_dims);
    assert_within_tolerance(
        &baseline_output,
        &optimized_output,
        "raw spinor kinetic optimizer on/off parity",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &baseline_output,
        "raw spinor kinetic optimizer-off vs wrapper",
    );
    assert_within_tolerance(
        &wrapper_scalars,
        &optimized_output,
        "raw spinor kinetic optimizer-on vs wrapper",
    );
}

fn kinetic_operator() -> Operator {
    Operator::new(IntegralFamily::OneElectron, OperatorKind::Kinetic)
        .expect("one-electron kinetic operator must be valid")
}

fn parity_options(extra_feature_flags: &[&'static str]) -> WorkspaceQueryOptions {
    phase2_cpu_options(extra_feature_flags)
}

fn evaluate_raw_spherical_kinetic(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    options: &WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spherical,
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
    .unwrap_or_else(|err| panic!("raw spherical kinetic query failed for {shls:?}: {err:?}"));

    let mut output = vec![0.0f64; workspace.required_bytes / 8];
    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spherical,
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
    .unwrap_or_else(|err| panic!("raw spherical kinetic evaluate failed for {shls:?}: {err:?}"));

    (workspace, output)
}

fn evaluate_raw_spinor_kinetic(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    options: &WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let workspace = raw::query_workspace_compat_with_sentinels(
        kinetic_operator(),
        Representation::Spinor,
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
    .unwrap_or_else(|err| panic!("raw spinor kinetic query failed for {shls:?}: {err:?}"));

    let mut output = vec![0.0f64; workspace.required_bytes / 8];
    raw::evaluate_compat(
        kinetic_operator(),
        Representation::Spinor,
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
    .unwrap_or_else(|err| panic!("raw spinor kinetic evaluate failed for {shls:?}: {err:?}"));

    (workspace, output)
}

fn wrapper_kinetic_spherical(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, CIntType::Spheric);
    let shls_slice = [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
    ];
    cint.integrate("int1e_kin", None, shls_slice).into()
}

fn wrapper_kinetic_spinor(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, CIntType::Spinor);
    let shls_slice = [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
    ];
    let (out, dims) = cint.integrate_spinor("int1e_kin", None, shls_slice).into();
    let mut flattened = Vec::with_capacity(out.len() * 2);
    for value in out {
        flattened.push(value.re);
        flattened.push(value.im);
    }
    (flattened, dims)
}

fn wrapper_cint_from_raw_layout(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    cint_type: CIntType,
) -> CInt {
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
        cint_type,
    }
}

fn flatten_real_output(output: EvaluationOutput) -> Vec<f64> {
    match output {
        EvaluationOutput::Real(values) => values,
        EvaluationOutput::Spinor(values) => panic!(
            "spherical kinetic must produce real output, got spinor len={}",
            values.len()
        ),
    }
}

fn flatten_spinor_output(output: EvaluationOutput) -> Vec<f64> {
    match output {
        EvaluationOutput::Spinor(values) => flatten_spinor_pairs(&values),
        EvaluationOutput::Real(values) => {
            panic!(
                "spinor kinetic must produce complex output, got real len={}",
                values.len()
            )
        }
    }
}

fn flatten_spinor_pairs(values: &[[f64; 2]]) -> Vec<f64> {
    let mut flattened = Vec::with_capacity(values.len() * 2);
    for value in values {
        flattened.push(value[0]);
        flattened.push(value[1]);
    }
    flattened
}

fn col_major_to_row_major_real(column_major: &[f64], dims: &[usize]) -> Vec<f64> {
    assert_eq!(dims.len(), 2, "1e real fixture must be rank-2");
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

fn col_major_to_row_major_spinor(column_major: &[f64], dims: &[usize]) -> Vec<f64> {
    assert_eq!(dims.len(), 2, "1e spinor fixture must be rank-2");
    let di = dims[0];
    let dj = dims[1];
    let mut row_major = vec![0.0f64; column_major.len()];
    for j in 0..dj {
        for i in 0..di {
            let col_element = i + di * j;
            let row_element = j + dj * i;
            row_major[2 * row_element] = column_major[2 * col_element];
            row_major[2 * row_element + 1] = column_major[2 * col_element + 1];
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

fn checked_product(dims: &[usize]) -> usize {
    let mut product = 1usize;
    for dim in dims {
        product = product
            .checked_mul(*dim)
            .expect("dimension product should fit usize for test fixtures");
    }
    product
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
