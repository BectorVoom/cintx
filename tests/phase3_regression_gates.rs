#[path = "common/oracle_runner.rs"]
mod oracle_runner;
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use std::collections::BTreeSet;

use cintx::{ManifestProfile, raw, safe};
use oracle_runner::{
    PHASE3_REQUIRED_GATE_REQUIREMENTS, TolerancePolicy, assert_requirement_traceability,
    assert_within_tolerance, oracle_expected_scalars_with_wrapper_override,
    phase3_oracle_profile_matrix,
};
use phase2_fixtures::{
    flatten_safe_output, phase2_cpu_options, stable_phase2_matrix, stable_raw_layout,
    stable_safe_basis,
};

#[test]
fn oracle_profile_matrix_gate() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();
    let matrix = stable_phase2_matrix();

    for profile_case in phase3_oracle_profile_matrix() {
        assert_requirement_traceability(
            profile_case.requirement_ids,
            PHASE3_REQUIRED_GATE_REQUIREMENTS,
            &format!("profile {}", profile_case.profile.as_str()),
        );
        let options = phase2_cpu_options(profile_case.feature_flags);
        let tolerance = TolerancePolicy::strict();

        for row in &matrix {
            let row = *row;
            let operator = row.operator();
            let row_id = row.id();

            let safe_tensor = safe::evaluate(
                &basis,
                operator,
                row.representation,
                row.safe_shell_tuple,
                &options,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "safe evaluate failed for profile {} row {row_id}: {err:?}",
                    profile_case.profile.as_str(),
                )
            });
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
            .unwrap_or_else(|err| {
                panic!(
                    "raw query failed for profile {} row {row_id}: {err:?}",
                    profile_case.profile.as_str(),
                )
            });

            let mut raw_output = vec![0.0f64; raw_workspace.required_bytes / 8];
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
                    out: &mut raw_output,
                    cache: None,
                    opt: None,
                },
                &options,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "raw evaluate failed for profile {} row {row_id}: {err:?}",
                    profile_case.profile.as_str(),
                )
            });

            let oracle_scalars = oracle_expected_scalars_with_wrapper_override(
                row.route_key(),
                row.representation,
                &raw_workspace.dims,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "oracle generation failed for profile {} row {row_id}: {err:?}",
                    profile_case.profile.as_str(),
                )
            });

            assert_eq!(safe_tensor.dims, raw_workspace.dims);
            assert_eq!(raw_result.dims, raw_workspace.dims);
            assert_eq!(safe_scalars.len(), oracle_scalars.len());
            assert_eq!(raw_output.len(), oracle_scalars.len());

            let context_prefix = format!(
                "profile={} row={row_id} requirements={:?}",
                profile_case.profile.as_str(),
                profile_case.requirement_ids,
            );
            assert_within_tolerance(
                &oracle_scalars,
                &safe_scalars,
                tolerance,
                &format!("{context_prefix} safe-vs-oracle"),
            );
            assert_within_tolerance(
                &oracle_scalars,
                &raw_output,
                tolerance,
                &format!("{context_prefix} raw-vs-oracle"),
            );
        }
    }
}

#[test]
fn requirement_traceability_gate() {
    let expected_profiles: BTreeSet<ManifestProfile> =
        ManifestProfile::approved_scope().into_iter().collect();
    let observed_profiles: BTreeSet<ManifestProfile> = phase3_oracle_profile_matrix()
        .iter()
        .map(|case| case.profile)
        .collect();
    assert_eq!(
        observed_profiles, expected_profiles,
        "profile-aware oracle matrix must cover the full approved profile scope",
    );

    let mut observed_requirements = BTreeSet::new();
    for case in phase3_oracle_profile_matrix() {
        assert_requirement_traceability(
            case.requirement_ids,
            PHASE3_REQUIRED_GATE_REQUIREMENTS,
            &format!("profile {}", case.profile.as_str()),
        );
        for requirement in case.requirement_ids {
            observed_requirements.insert(*requirement);
        }
    }
    let expected_requirements: BTreeSet<&str> =
        PHASE3_REQUIRED_GATE_REQUIREMENTS.iter().copied().collect();
    assert_eq!(
        observed_requirements, expected_requirements,
        "oracle regression gates must map exactly to COMP-04 and VERI-02 requirement IDs",
    );

    let matrix = stable_phase2_matrix();
    assert_eq!(
        matrix.len(),
        14,
        "profile-aware regression matrix must cover all currently supported stable envelopes",
    );
}
