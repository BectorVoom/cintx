use anyhow::{anyhow, bail, Context, Result};
use cintx_ops::resolver::{HelperKind, Resolver, Stability};
use cintx_oracle::fixtures::{
    build_profile_representation_matrix, build_required_profile_matrices, is_oracle_eligible_family,
    write_pretty_json_artifact, OracleRawInputs, PHASE4_APPROVED_PROFILES,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const COMPILED_MANIFEST_LOCK_JSON: &str = "crates/cintx-ops/generated/compiled_manifest.lock.json";
const REQUIRED_AUDIT_ARTIFACT: &str = "/mnt/data/cintx_phase_04_manifest_audit.json";
const AUDIT_ARTIFACT_FALLBACK_NAME: &str = "cintx_phase_04_manifest_audit.json";
const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const REQUIRED_PROFILE_CSV: &str = "base,with-f12,with-4c1e,with-f12+with-4c1e";

pub fn run_manifest_audit(profiles: &[String], check_lock: bool) -> Result<()> {
    let requested_profiles: BTreeSet<String> = profiles.iter().cloned().collect();
    let required_profiles: BTreeSet<String> = PHASE4_APPROVED_PROFILES
        .iter()
        .map(|profile| (*profile).to_owned())
        .collect();

    let profile_scope_mismatch = evaluate_profile_scope_mismatch(&requested_profiles, &required_profiles);
    let inputs = OracleRawInputs::sample();
    let required_matrices = build_required_profile_matrices(&inputs)?;
    let generated_required_profiles: BTreeSet<String> = required_matrices
        .iter()
        .map(|matrix| matrix.profile.clone())
        .collect();

    let lock_root = load_compiled_manifest_lock()?;
    let lock_approved_profiles = collect_profile_scope_values(&lock_root, "approved");
    let lock_observed_profiles = collect_profile_scope_values(&lock_root, "observed_union");

    let mut missing_in_lock: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut missing_in_generated: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let oracle_scope = oracle_scope_symbols_by_profile();

    for profile in PHASE4_APPROVED_PROFILES {
        let generated_symbols = collect_generated_symbols_for_profile(&inputs, profile)?;
        let lock_symbols = collect_lock_symbols_for_profile(
            &lock_root,
            profile,
            oracle_scope.get(*profile),
        )?;

        let missing_lock = set_difference(&generated_symbols, &lock_symbols);
        let missing_generated = set_difference(&lock_symbols, &generated_symbols);
        missing_in_lock.insert((*profile).to_owned(), missing_lock);
        missing_in_generated.insert((*profile).to_owned(), missing_generated);
    }

    let has_symbol_drift = missing_in_lock.values().any(|symbols| !symbols.is_empty())
        || missing_in_generated.values().any(|symbols| !symbols.is_empty());

    let has_profile_scope_mismatch = profile_scope_mismatch
        .get("has_mismatch")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let has_required_matrix_scope_mismatch = generated_required_profiles != required_profiles;

    let uncovered_stable = if check_lock {
        check_oracle_coverage(&lock_root)
    } else {
        Vec::new()
    };

    let mut report = json!({
        "compiled_manifest_lock": COMPILED_MANIFEST_LOCK_JSON,
        "required_path": REQUIRED_AUDIT_ARTIFACT,
        "profiles_requested": sorted_strings(requested_profiles.iter().cloned().collect()),
        "required_profiles": PHASE4_APPROVED_PROFILES,
        "required_profiles_csv": REQUIRED_PROFILE_CSV,
        "check_lock": check_lock,
        "missing_in_lock": missing_in_lock,
        "missing_in_generated": missing_in_generated,
        "profile_scope_mismatch": {
            "requested_vs_required": profile_scope_mismatch,
            "lock_approved_missing": set_difference(&required_profiles, &lock_approved_profiles),
            "lock_approved_extra": set_difference(&lock_approved_profiles, &required_profiles),
            "lock_observed_missing": set_difference(&required_profiles, &lock_observed_profiles),
            "lock_observed_extra": set_difference(&lock_observed_profiles, &required_profiles),
            "generated_matrix_missing_profiles": set_difference(&required_profiles, &generated_required_profiles),
            "generated_matrix_extra_profiles": set_difference(&generated_required_profiles, &required_profiles),
        },
        "artifact_policy": {
            "required_path": REQUIRED_AUDIT_ARTIFACT,
            "fallback_env_var": FALLBACK_ARTIFACT_DIR_ENV,
            "fallback_file_name": AUDIT_ARTIFACT_FALLBACK_NAME,
        },
        "oracle_coverage": {
            "uncovered_stable_entries": &uncovered_stable,
            "uncovered_count": uncovered_stable.len(),
        },
    });

    let should_fail = check_lock
        && (has_symbol_drift
            || has_profile_scope_mismatch
            || has_required_matrix_scope_mismatch
            || !uncovered_stable.is_empty());
    report["status"] = if should_fail {
        json!("failed")
    } else {
        json!("ok")
    };

    let report_path = write_manifest_audit_report(report)?;
    if should_fail {
        bail!(
            "manifest audit drift detected (see `{}`)",
            report_path.display()
        );
    }

    println!("manifest audit report: {}", report_path.display());
    Ok(())
}

fn evaluate_profile_scope_mismatch(
    requested_profiles: &BTreeSet<String>,
    required_profiles: &BTreeSet<String>,
) -> Value {
    let missing_profiles = set_difference(required_profiles, requested_profiles);
    let extra_profiles = set_difference(requested_profiles, required_profiles);
    json!({
        "missing_required_profiles": missing_profiles,
        "unexpected_profiles": extra_profiles,
        "has_mismatch": !set_difference(required_profiles, requested_profiles).is_empty()
            || !set_difference(requested_profiles, required_profiles).is_empty(),
    })
}

fn load_compiled_manifest_lock() -> Result<Value> {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../{COMPILED_MANIFEST_LOCK_JSON}"));
    let payload = fs::read_to_string(&lock_path)
        .with_context(|| format!("read compiled manifest lock `{}`", lock_path.display()))?;
    serde_json::from_str(&payload).context("parse compiled manifest lock json")
}

fn collect_profile_scope_values(lock_root: &Value, scope_key: &str) -> BTreeSet<String> {
    lock_root
        .get("profile_scope")
        .and_then(Value::as_object)
        .and_then(|profile_scope| profile_scope.get(scope_key))
        .and_then(Value::as_array)
        .map(|profiles| {
            profiles
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn collect_lock_symbols_for_profile(
    lock_root: &Value,
    profile: &str,
    oracle_scope: Option<&BTreeSet<String>>,
) -> Result<BTreeSet<String>> {
    let entries = lock_root
        .get("entries")
        .and_then(Value::as_array)
        .context("compiled manifest lock missing `entries` array")?;
    let mut symbols = BTreeSet::new();

    for entry in entries {
        let profiles = entry
            .get("profiles")
            .and_then(Value::as_array)
            .map(|profiles| {
                profiles
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if !profiles.contains(profile) {
            continue;
        }

        let stability = entry
            .get("stability")
            .and_then(Value::as_str)
            .unwrap_or("stable");
        if !stability_is_included(stability) {
            continue;
        }

        let id = entry
            .get("id")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("compiled manifest lock entry missing `id` object"))?;
        let family = id.get("family").and_then(Value::as_str).unwrap_or_default();
        if !is_phase4_oracle_family(family) {
            continue;
        }

        if let Some(symbol) = id.get("symbol").and_then(Value::as_str) {
            if let Some(scope_symbols) = oracle_scope {
                if !scope_symbols.contains(symbol) {
                    continue;
                }
            }
            symbols.insert(symbol.to_owned());
        }
    }

    Ok(symbols)
}

fn collect_generated_symbols_for_profile(
    inputs: &OracleRawInputs,
    profile: &str,
) -> Result<BTreeSet<String>> {
    let fixtures = build_profile_representation_matrix(inputs, profile, false)
        .with_context(|| format!("build profile representation matrix for `{profile}`"))?;
    Ok(fixtures
        .into_iter()
        .map(|fixture| fixture.symbol)
        .collect::<BTreeSet<_>>())
}

fn write_manifest_audit_report(mut report: Value) -> Result<PathBuf> {
    let artifact = write_pretty_json_artifact(
        REQUIRED_AUDIT_ARTIFACT,
        AUDIT_ARTIFACT_FALLBACK_NAME,
        &report,
    )?;
    report["artifact_write"] = json!({
        "required_path": artifact.required_path,
        "actual_path": artifact.actual_path.display().to_string(),
        "used_required_path": artifact.used_required_path,
        "fallback_reason": artifact.fallback_reason,
        "fallback_env_var": FALLBACK_ARTIFACT_DIR_ENV,
    });
    let payload = serde_json::to_vec_pretty(&report).context("serialize manifest audit report")?;
    fs::write(&artifact.actual_path, payload).with_context(|| {
        format!(
            "write manifest audit report `{}`",
            artifact.actual_path.display()
        )
    })?;
    Ok(artifact.actual_path)
}

fn check_oracle_coverage(lock_root: &Value) -> Vec<String> {
    let entries = match lock_root["entries"].as_array() {
        Some(e) => e,
        None => return Vec::new(),
    };
    let mut uncovered = Vec::new();
    for entry in entries {
        let stability = entry
            .get("stability")
            .and_then(Value::as_str)
            .unwrap_or("");
        if stability != "stable" {
            continue;
        }
        let covered = entry
            .get("oracle_covered")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !covered {
            let sym = entry
                .get("id")
                .and_then(|id| id.get("symbol"))
                .and_then(Value::as_str)
                .unwrap_or("?");
            uncovered.push(sym.to_owned());
        }
    }
    uncovered
}

fn stability_is_included(stability: &str) -> bool {
    matches!(stability, "stable" | "optional")
}

fn is_phase4_oracle_family(family: &str) -> bool {
    is_oracle_eligible_family(family)
}

fn set_difference(left: &BTreeSet<String>, right: &BTreeSet<String>) -> Vec<String> {
    sorted_strings(left.difference(right).cloned().collect())
}

fn oracle_scope_symbols_by_profile() -> BTreeMap<&'static str, BTreeSet<String>> {
    PHASE4_APPROVED_PROFILES
        .iter()
        .map(|profile| {
            let symbols = Resolver::manifest()
                .iter()
                .filter(|entry| matches!(entry.helper_kind, HelperKind::Operator | HelperKind::SourceOnly))
                .filter(|entry| is_phase4_oracle_family(entry.family_name))
                .filter(|entry| entry.is_compiled_in_profile(profile))
                .filter(|entry| !matches!(entry.stability, Stability::UnstableSource))
                .map(|entry| entry.symbol_name.to_owned())
                .collect::<BTreeSet<_>>();
            (*profile, symbols)
        })
        .collect()
}

fn sorted_strings(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values
}
