#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;
#[path = "common/phase3_helper_cases.rs"]
mod phase3_helper_cases;

use cintx::{
    deterministic_transform_scalars, route, safe, shell_ao_layout, shell_normalization_metadata,
    CpuRouteTarget, Representation,
};
use oracle_runner::{assert_within_tolerance, oracle_expected_scalars, TolerancePolicy};
use phase2_fixtures::{
    phase3_helper_options, stable_expected_shell_counts_cartesian,
    stable_expected_shell_counts_spherical, stable_expected_shell_counts_spinor,
    stable_expected_shell_offsets_cartesian, stable_expected_shell_offsets_spherical,
    stable_expected_shell_offsets_spinor, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};
use phase3_helper_cases::{
    expected_gto_norm, helper_matrix_case_count, stable_shell_normalization_expectations,
};

const NORM_TOLERANCE: f64 = 1e-12;

#[test]
fn helper_counts_offsets_normalization_parity() {
    let (_, bas, env) = stable_raw_layout();

    let cart_layout = shell_ao_layout(&bas, Representation::Cartesian)
        .expect("cartesian helper layout must be valid for stable fixture");
    assert_eq!(cart_layout.counts, stable_expected_shell_counts_cartesian());
    assert_eq!(
        cart_layout.offsets,
        stable_expected_shell_offsets_cartesian()
    );
    assert_eq!(cart_layout.total_count, 11);

    let sph_layout = shell_ao_layout(&bas, Representation::Spherical)
        .expect("spherical helper layout must be valid for stable fixture");
    assert_eq!(sph_layout.counts, stable_expected_shell_counts_spherical());
    assert_eq!(
        sph_layout.offsets,
        stable_expected_shell_offsets_spherical()
    );
    assert_eq!(sph_layout.total_count, 10);

    let spinor_layout = shell_ao_layout(&bas, Representation::Spinor)
        .expect("spinor helper layout must be valid for stable fixture");
    assert_eq!(spinor_layout.counts, stable_expected_shell_counts_spinor());
    assert_eq!(
        spinor_layout.offsets,
        stable_expected_shell_offsets_spinor()
    );
    assert_eq!(spinor_layout.total_count, 20);

    for expected in stable_shell_normalization_expectations() {
        let metadata = shell_normalization_metadata(expected.shell_index, &bas, &env)
            .unwrap_or_else(|err| panic!("normalization metadata failed: {err:?}"));
        assert_eq!(metadata.angular_momentum, expected.angular_momentum);
        assert_eq!(metadata.exponents, expected.exponents);
        assert_eq!(metadata.coefficients, expected.coefficients);
        assert_eq!(
            metadata.primitive_norms.len(),
            metadata.exponents.len(),
            "primitive norm count must track primitive exponent count",
        );

        for (index, norm) in metadata.primitive_norms.iter().enumerate() {
            let expected_norm =
                expected_gto_norm(metadata.angular_momentum, metadata.exponents[index]);
            assert_close(
                *norm,
                expected_norm,
                &format!("shell {} primitive {index}", expected.shell_index),
            );
            assert_close(
                metadata.normalized_coefficients[index],
                metadata.coefficients[index] * *norm,
                &format!(
                    "shell {} normalized coefficient {index}",
                    expected.shell_index
                ),
            );
        }
    }
}

#[test]
fn helper_transform_parity_matrix() {
    let basis = stable_safe_basis();
    let options = phase3_helper_options();
    let matrix = stable_phase2_matrix();
    assert_eq!(
        matrix.len(),
        helper_matrix_case_count(),
        "stable-family helper matrix size must remain deterministic",
    );
    assert!(
        matrix.iter().any(|row| row.is_explicit_3c1e_spinor()),
        "helper transform matrix must include explicit 3c1e spinor coverage",
    );

    for row in matrix {
        let operator = row.operator();
        let row_id = row.id();
        let query = safe::query_workspace(
            &basis,
            operator,
            row.representation,
            row.safe_shell_tuple,
            &options,
        )
        .unwrap_or_else(|err| panic!("safe query failed for {row_id}: {err:?}"));

        let route_target = route(row.route_key())
            .unwrap_or_else(|err| panic!("route failed for {row_id}: {err:?}"));

        let helper_scalars =
            deterministic_transform_scalars(route_target, row.representation, &query.dims)
                .unwrap_or_else(|err| panic!("helper transform failed for {row_id}: {err:?}"));

        let oracle_scalars =
            oracle_expected_scalars(row.route_key(), row.representation, &query.dims)
                .unwrap_or_else(|err| panic!("oracle helper failed for {row_id}: {err:?}"));

        assert_within_tolerance(
            &oracle_scalars,
            &helper_scalars,
            TolerancePolicy::strict(),
            &format!("helper transform parity {row_id}"),
        );

        if row.is_explicit_3c1e_spinor() {
            match route_target {
                CpuRouteTarget::ThreeCenterOneElectronSpinor(_) => {}
                other => panic!("3c1e spinor must route through adapter, got {other:?}"),
            }
            let has_negative_imag = helper_scalars
                .iter()
                .skip(1)
                .step_by(2)
                .any(|value| *value < 0.0);
            assert!(
                has_negative_imag,
                "3c1e spinor helper transform must preserve adapter-specific imaginary sign",
            );
        }
    }
}

fn assert_close(got: f64, expected: f64, context: &str) {
    let diff = (expected - got).abs();
    if diff <= NORM_TOLERANCE {
        return;
    }

    let scale = expected.abs().max(got.abs()).max(1.0);
    let rel = diff / scale;
    assert!(
        rel <= NORM_TOLERANCE,
        "{context}: expected={expected}, got={got}, abs_diff={diff}, rel_diff={rel}",
    );
}
