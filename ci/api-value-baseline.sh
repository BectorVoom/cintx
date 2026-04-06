#!/usr/bin/env bash
set -euo pipefail

# API baseline gate:
# - Exercise safe API behavior (`cintx-rs`) for each required profile.
# - Exercise raw API staging/value contract checks (`cintx-compat`) for each profile.
# Run in CPU mode so hosted CI runners do not depend on a physical GPU.

profiles_csv="${CINTX_REQUIRED_PROFILES:-base,with-f12,with-4c1e,with-f12+with-4c1e}"
IFS=',' read -r -a profiles <<< "${profiles_csv}"

run_profile_baseline() {
    local profile="$1"
    local -a feature_flags=()

    case "${profile}" in
        base)
            ;;
        with-f12)
            feature_flags=(--features with-f12)
            ;;
        with-4c1e)
            feature_flags=(--features with-4c1e)
            ;;
        with-f12+with-4c1e)
            feature_flags=(--features with-f12,with-4c1e)
            ;;
        *)
            echo "api-value-baseline: unsupported profile '${profile}'" >&2
            return 1
            ;;
    esac

    echo "api-value-baseline: running profile=${profile}"

    CINTX_BACKEND=cpu cargo test --locked -p cintx-rs "${feature_flags[@]}" --lib
    CINTX_BACKEND=cpu cargo test --locked -p cintx-compat "${feature_flags[@]}" \
        raw::tests::eval_raw_reads_staging_directly -- --exact
    CINTX_BACKEND=cpu cargo test --locked -p cintx-compat "${feature_flags[@]}" \
        raw::tests::query_workspace_raw_and_eval_raw_none_match_workspace_expectations -- --exact
}

for profile in "${profiles[@]}"; do
    run_profile_baseline "${profile}"
done

echo "api-value-baseline: completed all profiles"
