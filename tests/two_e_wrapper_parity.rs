#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    CpuRouteKey, EvaluationOutputMut, ExecutionRequest, IntegralFamily, LibcintRsError, Operator,
    OperatorKind, RawEvaluateRequest, RawQueryRequest, Representation, RouteSurface,
    WorkspaceQueryOptions, raw, resolve_capi_route, resolve_raw_route, resolve_route,
    resolve_safe_route, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{
    flatten_safe_output, phase2_cpu_options, phase3_optimizer_options, raw_optimizer_cache_len,
    stable_raw_layout, stable_safe_basis,
};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

const REDUCED_SAFE_SHLS: &[usize] = &[2, 3, 1, 0];
const REDUCED_RAW_SHLS: &[i32] = &[2, 3, 1, 0];
const ALT_RAW_SHLS_A: &[i32] = &[0, 1, 2, 3];
const ALT_RAW_SHLS_B: &[i32] = &[3, 2, 1, 0];

#[test]
fn two_e_safe_and_raw_match_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for representation in supported_representations() {
        let options = parity_options(representation, "two-e-safe-raw-wrapper");
        let operator = two_e_operator();

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!("safe evaluate failed for {representation:?} reduced 2e fixture: {err:?}")
        });
        let safe_scalars = flatten_safe_output(safe_tensor.output);

        let (wrapper_scalars, wrapper_dims) =
            wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(safe_tensor.dims, wrapper_dims);
        assert_eq!(safe_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &safe_scalars,
            &format!("safe 2e {representation:?} reduced case vs wrapper"),
        );

        let (workspace, raw_scalars) =
            evaluate_raw_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation, &options);
        assert_eq!(workspace.dims, wrapper_dims);
        assert_eq!(raw_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw 2e {representation:?} reduced case vs wrapper"),
        );
    }
}

#[test]
fn two_e_safe_evaluate_into_matches_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for representation in supported_representations() {
        let options = parity_options(representation, "two-e-safe-evaluate-into-wrapper");
        let operator = two_e_operator();
        let (_, wrapper_dims) = wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);
        let element_count = checked_product(&wrapper_dims);

        match representation {
            Representation::Cartesian | Representation::Spherical => {
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
                    panic!(
                        "safe evaluate_into failed for {representation:?} reduced 2e fixture: {err:?}"
                    )
                });
                let (wrapper_scalars, expected_dims) =
                    wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);
                assert_eq!(metadata.dims, expected_dims);
                assert_eq!(output.len(), wrapper_scalars.len());
                assert_within_tolerance(
                    &wrapper_scalars,
                    &output,
                    &format!("safe evaluate_into 2e {representation:?} vs wrapper"),
                );
            }
            Representation::Spinor => {
                let mut output = vec![[0.0f64; 2]; element_count];
                let metadata = safe::evaluate_into(
                    &basis,
                    operator,
                    representation,
                    REDUCED_SAFE_SHLS,
                    &options,
                    EvaluationOutputMut::Spinor(&mut output),
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "safe evaluate_into failed for {representation:?} reduced 2e fixture: {err:?}"
                    )
                });
                let flattened = flatten_spinor_pairs(output.as_slice());
                let (wrapper_scalars, expected_dims) =
                    wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);
                assert_eq!(metadata.dims, expected_dims);
                assert_eq!(flattened.len(), wrapper_scalars.len());
                assert_within_tolerance(
                    &wrapper_scalars,
                    &flattened,
                    "safe evaluate_into 2e spinor vs wrapper",
                );
            }
        }
    }
}

#[test]
fn two_e_raw_uses_libcint_column_major_layout() {
    let (atm, bas, env) = stable_raw_layout();

    for representation in supported_representations() {
        let options = parity_options(representation, "two-e-raw-layout");
        let (workspace, raw_scalars) =
            evaluate_raw_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation, &options);
        let (wrapper_scalars, wrapper_dims) =
            wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(workspace.dims, wrapper_dims);
        let row_major_scalars = match representation {
            Representation::Cartesian | Representation::Spherical => {
                col_major_to_row_major_real(&wrapper_scalars, &wrapper_dims)
            }
            Representation::Spinor => col_major_to_row_major_spinor(&wrapper_scalars, &wrapper_dims),
        };
        assert_ne!(
            wrapper_scalars, row_major_scalars,
            "fixture must distinguish column-major and row-major ordering for {representation:?}"
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw 2e {representation:?} should preserve libcint flat column-major layout"),
        );
        let row_major_max_diff = max_abs_diff(&row_major_scalars, &raw_scalars);
        assert!(
            row_major_max_diff > 1e-9,
            "raw 2e {representation:?} unexpectedly matches row-major ordering: max_abs_diff={row_major_max_diff}",
        );
    }
}

#[test]
fn two_e_raw_interprets_atm_bas_env_and_shls_like_wrapper() {
    let (atm, bas, env) = stable_raw_layout();

    for representation in supported_representations() {
        let options = parity_options(representation, "two-e-raw-inputs");
        for shls in [ALT_RAW_SHLS_A, REDUCED_RAW_SHLS, ALT_RAW_SHLS_B] {
            let (workspace, raw_scalars) =
                evaluate_raw_two_e(&atm, &bas, &env, shls, representation, &options);
            let (wrapper_scalars, wrapper_dims) = wrapper_two_e(&atm, &bas, &env, shls, representation);

            assert_eq!(workspace.dims, wrapper_dims, "dims drift for shls={shls:?}");
            assert_within_tolerance(
                &wrapper_scalars,
                &raw_scalars,
                &format!("raw 2e {representation:?} shell interpretation shls={shls:?}"),
            );
        }
    }
}

#[test]
fn two_e_raw_optimizer_mode_is_invariant_and_wrapper_aligned() {
    let (atm, bas, env) = stable_raw_layout();

    for representation in supported_representations() {
        let baseline_options = phase3_optimizer_options(&["two-e-optimizer-off"]);
        let optimized_options = phase3_optimizer_options(&["two-e-optimizer-on"]);
        let operator = two_e_operator();

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

        let optimizer_query_cache = vec![0.0f64; raw_optimizer_cache_len(REDUCED_RAW_SHLS)];
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
                cache: Some(optimizer_query_cache.as_slice()),
                opt: Some(NonNull::<c_void>::dangling()),
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
        .unwrap_or_else(|err| panic!("baseline raw evaluate failed for {representation:?}: {err:?}"));

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
                cache: Some(optimized_cache.as_mut_slice()),
                opt: Some(NonNull::<c_void>::dangling()),
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| panic!("optimized raw evaluate failed for {representation:?}: {err:?}"));

        let (wrapper_scalars, wrapper_dims) =
            wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
        assert_eq!(baseline_workspace.dims, wrapper_dims);
        assert_within_tolerance(
            &baseline_output,
            &optimized_output,
            &format!("raw 2e {representation:?} optimizer on/off invariance"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &baseline_output,
            &format!("raw 2e {representation:?} optimizer-off vs wrapper"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &optimized_output,
            &format!("raw 2e {representation:?} optimizer-on vs wrapper"),
        );
    }
}

#[test]
fn two_e_query_and_evaluate_surfaces_follow_shared_route_policy() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["two-e-route-policy"]);
    let operator = two_e_operator();

    for representation in supported_representations() {
        let route_key = CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        );
        let expected_route_id = route_id_for_representation(representation);
        let request = ExecutionRequest::from_safe(operator, representation, REDUCED_SAFE_SHLS, &options);

        let safe_route = resolve_safe_route(&request)
            .unwrap_or_else(|err| panic!("safe resolve failed for {representation:?}: {err:?}"));
        let raw_route = resolve_raw_route(&request)
            .unwrap_or_else(|err| panic!("raw resolve failed for {representation:?}: {err:?}"));
        let capi_route = resolve_capi_route(&request)
            .unwrap_or_else(|err| panic!("capi resolve failed for {representation:?}: {err:?}"));

        assert_eq!(safe_route.route_id, expected_route_id);
        assert_eq!(raw_route.route_id, expected_route_id);
        assert_eq!(capi_route.route_id, expected_route_id);

        let safe_query = safe::query_workspace(&basis, operator, representation, REDUCED_SAFE_SHLS, &options)
            .unwrap_or_else(|err| panic!("safe query failed for {representation:?}: {err:?}"));
        let safe_eval = safe::evaluate(&basis, operator, representation, REDUCED_SAFE_SHLS, &options)
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

        assert_eq!(route_key, safe_route.key);
        assert_eq!(route_key, raw_route.key);
        assert_eq!(raw_result.dims, raw_query.dims);
    }
}

#[test]
fn two_e_unsupported_policy_route_is_blocked_by_shared_resolver() {
    assert!(
        Operator::new(IntegralFamily::TwoElectron, OperatorKind::Overlap).is_err(),
        "typed operator construction must reject unsupported 2e overlap pair",
    );

    let unsupported_key = CpuRouteKey::new(
        IntegralFamily::TwoElectron,
        OperatorKind::Overlap,
        Representation::Spherical,
    );
    for surface in [RouteSurface::Safe, RouteSurface::Raw, RouteSurface::CAbi] {
        let err = resolve_route(unsupported_key, surface)
            .expect_err("unsupported 2e route should fail in shared resolver");
        assert!(
            matches!(
                err,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "expected UnsupportedApi for unsupported 2e route on {surface:?}, got {err:?}",
        );
    }
}

#[test]
fn regression_two_e_cartesian_reduced_case_matches_wrapper_reference() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["two-e-regression-cartesian"]);

    let safe_tensor = safe::evaluate(
        &basis,
        two_e_operator(),
        Representation::Cartesian,
        REDUCED_SAFE_SHLS,
        &options,
    )
    .expect("safe Cartesian two-electron regression case must evaluate");
    let safe_scalars = flatten_safe_output(safe_tensor.output);

    let (wrapper_scalars, wrapper_dims) =
        wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, Representation::Cartesian);
    assert_eq!(safe_tensor.dims, wrapper_dims);
    assert_within_tolerance(
        &wrapper_scalars,
        &safe_scalars,
        "regression: Cartesian 2e reduced case must match wrapper after synthetic filler removal",
    );
}

#[test]
fn reproduces_pre_fix_seeded_filler_mismatch_for_cartesian_case() {
    let (atm, bas, env) = stable_raw_layout();
    let (wrapper_scalars, wrapper_dims) =
        wrapper_two_e(&atm, &bas, &env, REDUCED_RAW_SHLS, Representation::Cartesian);
    let legacy_scalars = legacy_seeded_real_scalars("int2e_cart", &wrapper_dims);

    let max_diff = max_abs_diff(&wrapper_scalars, &legacy_scalars);
    assert!(
        max_diff > 1e-3,
        "legacy seeded filler unexpectedly matched wrapper for reduced Cartesian 2e case",
    );
}

fn evaluate_raw_two_e(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
    options: &WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let workspace = raw::query_workspace_compat_with_sentinels(
        two_e_operator(),
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
    .unwrap_or_else(|err| panic!("raw query failed for 2e {representation:?} {shls:?}: {err:?}"));

    let mut output = vec![0.0f64; workspace.required_bytes / core::mem::size_of::<f64>()];
    raw::evaluate_compat(
        two_e_operator(),
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
    .unwrap_or_else(|err| panic!("raw evaluate failed for 2e {representation:?} {shls:?}: {err:?}"));

    (workspace, output)
}

fn wrapper_two_e(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, representation);
    let shls_slice = shell_ranges(shls);

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let (values, dims) = cint.integrate("int2e", None, shls_slice).into();
            (values, dims)
        }
        Representation::Spinor => {
            let (values, dims) = cint.integrate_spinor("int2e", None, shls_slice).into();
            let mut flattened = Vec::with_capacity(values.len() * 2);
            for value in values {
                flattened.push(value.re);
                flattened.push(value.im);
            }
            (flattened, dims)
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

fn shell_ranges(shls: &[i32]) -> [[usize; 2]; 4] {
    [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
        [shls[2] as usize, shls[2] as usize + 1],
        [shls[3] as usize, shls[3] as usize + 1],
    ]
}

fn parity_options(representation: Representation, gate: &'static str) -> WorkspaceQueryOptions {
    let label = match representation {
        Representation::Cartesian => "cart",
        Representation::Spherical => "sph",
        Representation::Spinor => "spinor",
    };
    phase2_cpu_options(&[gate, label])
}

fn two_e_operator() -> Operator {
    Operator::new(IntegralFamily::TwoElectron, OperatorKind::ElectronRepulsion)
        .expect("two-electron ERI operator should be valid")
}

fn supported_representations() -> [Representation; 3] {
    [
        Representation::Cartesian,
        Representation::Spherical,
        Representation::Spinor,
    ]
}

fn route_id_for_representation(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "int2e_eri.cart.cpu.direct.v1",
        Representation::Spherical => "int2e_eri.sph.cpu.direct.v1",
        Representation::Spinor => "int2e_eri.spinor.cpu.direct.v1",
    }
}

fn col_major_to_row_major_real(values: &[f64], dims: &[usize]) -> Vec<f64> {
    assert_eq!(dims.len(), 4, "2e layouts must be rank-4");
    let [d0, d1, d2, d3]: [usize; 4] = dims.try_into().expect("rank-4 dims required");
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

fn col_major_to_row_major_spinor(values: &[f64], dims: &[usize]) -> Vec<f64> {
    assert!(values.len().is_multiple_of(2));
    let pairs = values
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect::<Vec<_>>();
    let [d0, d1, d2, d3]: [usize; 4] = dims.try_into().expect("rank-4 dims required");
    let mut row_pairs = vec![[0.0f64; 2]; pairs.len()];

    for l in 0..d3 {
        for k in 0..d2 {
            for j in 0..d1 {
                for i in 0..d0 {
                    let col_index = i + d0 * (j + d1 * (k + d2 * l));
                    let row_index = ((i * d1 + j) * d2 + k) * d3 + l;
                    row_pairs[row_index] = pairs[col_index];
                }
            }
        }
    }

    flatten_spinor_pairs(row_pairs.as_slice())
}

fn flatten_spinor_pairs(values: &[[f64; 2]]) -> Vec<f64> {
    let mut flattened = Vec::with_capacity(values.len() * 2);
    for pair in values {
        flattened.push(pair[0]);
        flattened.push(pair[1]);
    }
    flattened
}

fn checked_product(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn max_abs_diff(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn legacy_seeded_real_scalars(route_symbol: &str, dims: &[usize]) -> Vec<f64> {
    let mut seed = 0u64;
    for byte in route_symbol.bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
    }
    for dim in dims {
        let dim_u64 = u64::try_from(*dim).unwrap_or(u64::MAX);
        seed = seed.wrapping_mul(257).wrapping_add(dim_u64);
    }

    let mut output = vec![0.0f64; checked_product(dims)];
    for (index, value) in output.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(17));
        *value = f64::from((raw % 4096) as u16) / 128.0;
    }
    output
}

fn assert_within_tolerance(expected: &[f64], actual: &[f64], context: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected and actual scalar lengths must match",
    );

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let diff = (expected_value - actual_value).abs();
        if diff <= ABS_TOLERANCE {
            continue;
        }

        let scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        let relative = diff / scale;
        assert!(
            relative <= REL_TOLERANCE,
            "{context}: mismatch at index {index}: expected={expected_value}, got={actual_value}, abs_diff={diff}, rel_diff={relative}",
        );
    }
}
