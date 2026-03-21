#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use core::ffi::c_void;
use core::ptr::NonNull;

use cintx::{
    CpuRouteKey, ExecutionRequest, IntegralFamily, Operator, OperatorKind, RawEvaluateRequest,
    RawQueryRequest, Representation, RouteStatus, raw, resolve_capi_route, resolve_raw_route,
    resolve_safe_route, route_manifest_entries, safe,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::{
    phase2_cpu_options, phase3_optimizer_options, stable_raw_layout, stable_safe_basis,
};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

const REDUCED_SAFE_SHLS: &[usize] = &[0, 1];
const REDUCED_RAW_SHLS: &[i32] = &[0, 1];

const SUPPORTED_REPRESENTATIONS: &[Representation] = &[
    Representation::Cartesian,
    Representation::Spherical,
    Representation::Spinor,
];

#[test]
fn two_c_route_inventory_uses_plain_2c2e_kernels() {
    let entries = route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.key.family == IntegralFamily::TwoCenterTwoElectron)
        .collect::<Vec<_>>();

    assert_eq!(
        entries.len(),
        3,
        "2c2e route inventory must include exactly three implemented rows",
    );
    assert!(
        entries
            .iter()
            .all(|entry| entry.status == RouteStatus::Implemented),
        "all 2c2e manifest rows must remain implemented",
    );

    let route_ids = entries
        .iter()
        .map(|entry| entry.route_id)
        .collect::<std::collections::BTreeSet<_>>();
    for expected in [
        "int2c2e_eri.cart.cpu.direct.v1",
        "int2c2e_eri.sph.cpu.direct.v1",
        "int2c2e_eri.spinor.cpu.direct.v1",
    ] {
        assert!(
            route_ids.contains(expected),
            "missing 2c2e manifest route `{expected}`",
        );
    }
    for entry in entries {
        let entry_kernel = entry.entry_kernel.as_str();
        assert!(
            !entry_kernel.contains("ip1"),
            "plain 2c2e route cannot use derivative ip1 kernel: {entry_kernel}",
        );
    }
}

#[test]
fn two_c_safe_and_raw_match_wrapper_for_supported_representations() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let operator = two_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "two-c-safe-raw-wrapper");

        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!("safe evaluate failed for 2c2e {representation:?} reduced fixture: {err:?}")
        });
        let safe_scalars = flatten_safe_output(safe_tensor.output);

        let (wrapper_scalars, wrapper_dims) =
            wrapper_two_c2e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);
        assert_eq!(safe_tensor.dims, wrapper_dims);
        assert_eq!(safe_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &safe_scalars,
            &format!("safe 2c2e {representation:?} reduced case vs wrapper"),
        );

        let (workspace, raw_scalars) =
            evaluate_raw_two_c2e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation, &options);
        assert_eq!(workspace.dims, wrapper_dims);
        assert_eq!(raw_scalars.len(), wrapper_scalars.len());
        assert_within_tolerance(
            &wrapper_scalars,
            &raw_scalars,
            &format!("raw 2c2e {representation:?} reduced case vs wrapper"),
        );
    }
}

#[test]
fn two_c_raw_optimizer_mode_is_invariant_and_wrapper_aligned() {
    let (atm, bas, env) = stable_raw_layout();
    let operator = two_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let baseline_options = phase3_optimizer_options(&["two-c-optimizer-off"]);
        let optimized_options = phase3_optimizer_options(&["two-c-optimizer-on"]);

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

        let optimizer_handle =
            wrapper_cint_from_raw_layout(&atm, &bas, &env, representation).optimizer("int2c2e");
        let optimizer_ptr =
            NonNull::new(optimizer_handle.as_ptr() as *mut c_void).expect("optimizer pointer");
        let optimizer_query_cache = vec![0.0f64; baseline_workspace.cache_required_len];

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
                opt: Some(optimizer_ptr),
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
        .unwrap_or_else(|err| {
            panic!("baseline raw evaluate failed for {representation:?}: {err:?}")
        });

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
                opt: Some(optimizer_ptr),
            },
            &optimized_options,
        )
        .unwrap_or_else(|err| {
            panic!("optimized raw evaluate failed for {representation:?}: {err:?}")
        });

        let (wrapper_scalars, wrapper_dims) =
            wrapper_two_c2e(&atm, &bas, &env, REDUCED_RAW_SHLS, representation);

        assert_eq!(baseline_workspace.dims, optimized_workspace.dims);
        assert_eq!(baseline_workspace.dims, wrapper_dims);
        assert!(
            optimized_workspace.cache_required_len >= baseline_workspace.cache_required_len,
            "optimized cache contract cannot shrink for {representation:?}",
        );
        assert_within_tolerance(
            &baseline_output,
            &optimized_output,
            &format!("raw 2c2e {representation:?} optimizer on/off invariance"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &baseline_output,
            &format!("raw 2c2e {representation:?} optimizer-off vs wrapper"),
        );
        assert_within_tolerance(
            &wrapper_scalars,
            &optimized_output,
            &format!("raw 2c2e {representation:?} optimizer-on vs wrapper"),
        );

        drop(optimizer_handle);
    }
}

#[test]
fn two_c_supported_routes_are_not_legacy_seeded_fallback() {
    let basis = stable_safe_basis();
    let operator = two_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let options = parity_options(representation, "two-c-regression-no-filler");
        let safe_tensor = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe evaluate failed for {representation:?}: {err:?}"));
        let dims = safe_tensor.dims;
        let actual = flatten_safe_output(safe_tensor.output);
        let legacy =
            legacy_seeded_real_scalars(entry_symbol_for_representation(representation), &dims);
        let max_diff = max_abs_diff(&legacy, &actual);
        assert!(
            max_diff > 1e-9,
            "2c2e {representation:?} output still matches legacy seeded filler; max_abs_diff={max_diff}",
        );
    }
}

#[test]
fn two_c_query_and_evaluate_surfaces_follow_shared_route_policy() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let options = phase2_cpu_options(&["two-c-route-policy"]);
    let operator = two_c_operator();

    for &representation in SUPPORTED_REPRESENTATIONS {
        let expected_route_id = route_id_for_representation(representation);
        let route_key = CpuRouteKey::new(
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            representation,
        );
        let request =
            ExecutionRequest::from_safe(operator, representation, REDUCED_SAFE_SHLS, &options);

        let safe_route = resolve_safe_route(&request)
            .unwrap_or_else(|err| panic!("safe resolve failed for {representation:?}: {err:?}"));
        let raw_route = resolve_raw_route(&request)
            .unwrap_or_else(|err| panic!("raw resolve failed for {representation:?}: {err:?}"));
        let capi_route = resolve_capi_route(&request)
            .unwrap_or_else(|err| panic!("capi resolve failed for {representation:?}: {err:?}"));

        assert_eq!(safe_route.route_id, expected_route_id);
        assert_eq!(raw_route.route_id, expected_route_id);
        assert_eq!(capi_route.route_id, expected_route_id);
        assert!(
            !safe_route.entry_kernel.as_str().contains("ip1"),
            "plain 2c2e safe route cannot dispatch to derivative kernels",
        );

        let safe_query = safe::query_workspace(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe query failed for {representation:?}: {err:?}"));
        let safe_eval = safe::evaluate(
            &basis,
            operator,
            representation,
            REDUCED_SAFE_SHLS,
            &options,
        )
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

        assert_eq!(safe_route.key, route_key);
        assert_eq!(raw_route.key, route_key);
        assert_eq!(capi_route.key, route_key);
        assert_eq!(safe_query.dims, raw_query.dims);
        assert_eq!(raw_query.dims, raw_result.dims);
    }
}

fn evaluate_raw_two_c2e(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
    options: &cintx::WorkspaceQueryOptions,
) -> (cintx::RawCompatWorkspace, Vec<f64>) {
    let operator = two_c_operator();
    let workspace = raw::query_workspace_compat_with_sentinels(
        operator,
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
    .unwrap_or_else(|err| panic!("raw query failed for 2c2e/{representation:?} {shls:?}: {err:?}"));

    let mut output = vec![0.0f64; workspace.required_bytes / core::mem::size_of::<f64>()];
    raw::evaluate_compat(
        operator,
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
    .unwrap_or_else(|err| {
        panic!("raw evaluate failed for 2c2e/{representation:?} {shls:?}: {err:?}")
    });

    (workspace, output)
}

fn wrapper_two_c2e(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32],
    representation: Representation,
) -> (Vec<f64>, Vec<usize>) {
    let cint = wrapper_cint_from_raw_layout(atm, bas, env, representation);
    let shls_slice = shell_ranges_2(shls);

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let (values, dims) = cint.integrate("int2c2e", None, shls_slice).into();
            (values, dims)
        }
        Representation::Spinor => {
            let (values, dims) = cint.integrate_spinor("int2c2e", None, shls_slice).into();
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

fn shell_ranges_2(shls: &[i32]) -> [[usize; 2]; 2] {
    [
        [shls[0] as usize, shls[0] as usize + 1],
        [shls[1] as usize, shls[1] as usize + 1],
    ]
}

fn parity_options(
    representation: Representation,
    gate: &'static str,
) -> cintx::WorkspaceQueryOptions {
    phase2_cpu_options(&[gate, representation_label(representation)])
}

fn representation_label(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "cart",
        Representation::Spherical => "sph",
        Representation::Spinor => "spinor",
    }
}

fn two_c_operator() -> Operator {
    Operator::new(
        IntegralFamily::TwoCenterTwoElectron,
        OperatorKind::ElectronRepulsion,
    )
    .expect("2c2e electron-repulsion operator should be valid")
}

fn route_id_for_representation(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "int2c2e_eri.cart.cpu.direct.v1",
        Representation::Spherical => "int2c2e_eri.sph.cpu.direct.v1",
        Representation::Spinor => "int2c2e_eri.spinor.cpu.direct.v1",
    }
}

fn entry_symbol_for_representation(representation: Representation) -> &'static str {
    match representation {
        Representation::Cartesian => "int2c2e_cart",
        Representation::Spherical => "int2c2e_sph",
        Representation::Spinor => "int2c2e_spinor",
    }
}

fn flatten_safe_output(output: cintx::EvaluationOutput) -> Vec<f64> {
    match output {
        cintx::EvaluationOutput::Real(values) => values,
        cintx::EvaluationOutput::Spinor(values) => {
            let mut flattened = Vec::with_capacity(values.len() * 2);
            for value in values {
                flattened.push(value[0]);
                flattened.push(value[1]);
            }
            flattened
        }
    }
}

fn assert_within_tolerance(expected: &[f64], actual: &[f64], context: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected and actual lengths differ",
    );
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let abs_diff = (expected_value - actual_value).abs();
        if abs_diff <= ABS_TOLERANCE {
            continue;
        }
        let scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        let rel_diff = abs_diff / scale;
        assert!(
            rel_diff <= REL_TOLERANCE,
            "{context}: mismatch at index {index}: expected={expected_value}, actual={actual_value}, abs_diff={abs_diff}, rel_diff={rel_diff}",
        );
    }
}

fn legacy_seeded_real_scalars(route_symbol: &str, dims: &[usize]) -> Vec<f64> {
    let mut seed = 0u64;
    for byte in route_symbol.bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
    }
    for &dim in dims {
        let dim_u64 = u64::try_from(dim).unwrap_or(u64::MAX);
        seed = seed.wrapping_mul(257).wrapping_add(dim_u64);
    }

    let len = checked_product(dims);
    let mut out = vec![0.0f64; len];
    for (index, value) in out.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(17));
        *value = f64::from((raw % 4096) as u16) / 128.0;
    }
    out
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f64, f64::max)
}

fn checked_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .expect("dimension product should fit usize in test fixtures")
}
