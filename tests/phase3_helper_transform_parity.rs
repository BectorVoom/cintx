#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;
#[path = "common/phase3_helper_cases.rs"]
mod phase3_helper_cases;

use cintx::{
    Representation, deterministic_transform_scalars, gto_norm, route, safe, shell_ao_layout,
    shell_normalization_metadata,
};
use oracle_runner::{TolerancePolicy, assert_within_tolerance, oracle_expected_scalars};
use phase2_fixtures::{
    out_of_phase_route_keys, phase3_helper_options, stable_expected_shell_counts_cartesian,
    stable_expected_shell_counts_spherical, stable_expected_shell_counts_spinor,
    stable_expected_shell_offsets_cartesian, stable_expected_shell_offsets_spherical,
    stable_expected_shell_offsets_spinor, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};
use phase3_helper_cases::{
    expected_gto_norm, helper_matrix_case_count, malformed_positive_kappa_bas,
    malformed_truncated_bas, stable_shell_normalization_expectations,
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
    }
}

#[test]
fn helper_failure_semantics_are_typed() {
    let truncated_bas = malformed_truncated_bas();
    let truncated_bas_err = shell_ao_layout(&truncated_bas, Representation::Cartesian)
        .expect_err("truncated bas rows must fail helper layout validation");
    assert!(matches!(
        truncated_bas_err,
        cintx::LibcintRsError::InvalidInput { field: "bas", reason }
            if reason.contains("not divisible by BAS_SLOTS")
    ));

    let malformed_spinor_bas = malformed_positive_kappa_bas();
    let malformed_spinor_err = shell_ao_layout(&malformed_spinor_bas, Representation::Spinor)
        .expect_err("positive kappa on l=0 shell must fail helper spinor counts");
    assert!(matches!(
        malformed_spinor_err,
        cintx::LibcintRsError::InvalidInput {
            field: "shell.kappa",
            reason
        } if reason.contains("positive kappa requires angular momentum > 0")
    ));

    let route_target = route(
        stable_phase2_matrix()
            .first()
            .expect("stable helper matrix cannot be empty")
            .route_key(),
    )
    .expect("stable helper matrix route must resolve");
    let dims_err =
        deterministic_transform_scalars(route_target, Representation::Cartesian, &[2, 0])
            .expect_err("zero dimension must fail helper transform scalar generation");
    assert!(matches!(
        dims_err,
        cintx::LibcintRsError::InvalidInput {
            field: "dims",
            reason
        } if reason.contains("must be greater than zero")
    ));

    let exponent_err = gto_norm(2, 0.0).expect_err("non-positive exponent must fail normalization");
    assert!(matches!(
        exponent_err,
        cintx::LibcintRsError::InvalidInput {
            field: "exponent",
            reason
        } if reason.contains("greater than zero")
    ));

    let (_, bas, mut env) = stable_raw_layout();
    env.truncate(33);
    let env_err = shell_normalization_metadata(1, &bas, &env)
        .expect_err("short env buffer must fail offset-based normalization metadata");
    assert!(matches!(
        env_err,
        cintx::LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason
        } if reason.contains("exceeds env length")
    ));

    for route_key in out_of_phase_route_keys() {
        let route_err = route(route_key)
            .expect_err("out-of-phase helper route must fail with typed unsupported");
        assert!(matches!(
            route_err,
            cintx::LibcintRsError::UnsupportedApi {
                api: "cpu.route",
                ..
            }
        ));
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
