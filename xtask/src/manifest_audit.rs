use anyhow::{Context, Result, anyhow, bail};
use cintx_ops::resolver::{HelperKind, Resolver, Stability};
use cintx_oracle::fixtures::{
    OracleRawInputs, PHASE4_APPROVED_PROFILES, build_profile_representation_matrix,
    build_required_profile_matrices, is_dedicated_oracle_family, is_oracle_eligible_family,
    write_pretty_json_artifact,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const COMPILED_MANIFEST_LOCK_JSON: &str = "crates/cintx-ops/generated/compiled_manifest.lock.json";
const REQUIRED_AUDIT_ARTIFACT: &str = "/tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json";
const AUDIT_ARTIFACT_FALLBACK_NAME: &str = "cintx_phase_04_manifest_audit.json";
const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const REQUIRED_PROFILE_CSV: &str = "base,with-f12,with-4c1e,with-f12+with-4c1e";

pub fn run_manifest_audit(profiles: &[String], check_lock: bool) -> Result<()> {
    let requested_profiles: BTreeSet<String> = profiles.iter().cloned().collect();
    let required_profiles: BTreeSet<String> = PHASE4_APPROVED_PROFILES
        .iter()
        .map(|profile| (*profile).to_owned())
        .collect();

    let profile_scope_mismatch =
        evaluate_profile_scope_mismatch(&requested_profiles, &required_profiles);
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
        let lock_symbols =
            collect_lock_symbols_for_profile(&lock_root, profile, oracle_scope.get(*profile))?;

        let missing_lock = set_difference(&generated_symbols, &lock_symbols);
        let missing_generated = set_difference(&lock_symbols, &generated_symbols);
        missing_in_lock.insert((*profile).to_owned(), missing_lock);
        missing_in_generated.insert((*profile).to_owned(), missing_generated);
    }

    let has_symbol_drift = missing_in_lock.values().any(|symbols| !symbols.is_empty())
        || missing_in_generated
            .values()
            .any(|symbols| !symbols.is_empty());

    let has_profile_scope_mismatch = profile_scope_mismatch
        .get("has_mismatch")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let has_required_matrix_scope_mismatch = generated_required_profiles != required_profiles;

    let coverage = if check_lock {
        check_oracle_coverage(&lock_root)
    } else {
        OracleCoverageBuckets::default()
    };
    let uncovered_stable = &coverage.uncovered;
    // W5-00: a self-contradicting row (covered AND fail-closed) is always a
    // failure, whether or not --check-lock was requested.
    let policy_contradictions = check_policy_contradictions(&lock_root);

    // PARITY-01: libcint's public surface is wider than cint_funcs.h.  PySCF
    // reaches these compiled wrappers by dynamic symbol lookup, so scan the
    // actual ALL_CINT/ALL_CINT1E invocations as a third reference source.
    let exported_libcint_families = collect_all_cint_exports()?;
    let manifest_libcint_families = collect_manifest_base_symbols();
    let unsupported_libcint_families =
        set_difference(&exported_libcint_families, &manifest_libcint_families);
    let parity_strict = std::env::var("CINTX_PARITY_STRICT").as_deref() == Ok("1");
    // W5-07: the baseline correspondence is checked unconditionally.
    let parity_baseline_drift = evaluate_parity_baseline(&unsupported_libcint_families);

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
            "uncovered_stable_entries": uncovered_stable,
            "uncovered_count": uncovered_stable.len(),
            "fail_closed_entries": &coverage.fail_closed,
            "fail_closed_count": coverage.fail_closed.len(),
            "no_upstream_oracle_entries": &coverage.no_upstream_oracle,
            "no_upstream_oracle_count": coverage.no_upstream_oracle.len(),
            "policy_contradictions": &policy_contradictions,
        },
        "libcint_export_parity": {
            "reference": "libcint-master/src/**/*.c ALL_CINT/ALL_CINT1E invocations",
            "strict": parity_strict,
            "exported_count": exported_libcint_families.len(),
            "manifest_base_count": manifest_libcint_families.len(),
            "unsupported_libcint_families": &unsupported_libcint_families,
            "unsupported_count": unsupported_libcint_families.len(),
            "baseline_dated": "2026-08-22",
            "baseline_count": PARITY_BASELINE.len(),
            "baseline_drift": &parity_baseline_drift,
        },
    });

    let should_fail = !policy_contradictions.is_empty()
        || !parity_baseline_drift.is_empty()
        || (check_lock
            && (has_symbol_drift
                || has_profile_scope_mismatch
                || has_required_matrix_scope_mismatch
                || !uncovered_stable.is_empty()
                || (parity_strict && !unsupported_libcint_families.is_empty())));
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

fn collect_manifest_base_symbols() -> BTreeSet<String> {
    Resolver::manifest()
        .iter()
        .filter_map(|entry| {
            entry
                .symbol_name
                .strip_suffix("_cart")
                .or_else(|| entry.symbol_name.strip_suffix("_sph"))
                .or_else(|| entry.symbol_name.strip_suffix("_spinor"))
        })
        .map(ToOwned::to_owned)
        .collect()
}

/// Wave 5 W5-07 — the PARITY-01 baseline.
///
/// `CINTX_PARITY_STRICT=1` demands `unsupported_libcint_families` be *empty*,
/// which is Phase 31's exit gate, not Wave 5's: 43 of the 52 unsupported symbols
/// legitimately belong to Phases 30 and 31. A permanently-red strict gate trains
/// everyone to ignore it.
///
/// Instead every currently-unsupported symbol is enumerated here with the phase
/// that owns closing it. The audit then fails, unconditionally and with no env
/// var, if either half of the correspondence breaks:
///
///   * a symbol is unsupported but NOT on this list — a new Gap-A-shaped omission,
///     which is the defect class PARITY-01 actually exists to catch;
///   * a symbol is on this list but NO LONGER unsupported — a stale entry, which
///     forces the list to shrink monotonically as families land.
///
/// Dated 2026-08-22. Wave 5 removes its own nine as W5-05/W5-06 land.
const PARITY_BASELINE: &[(&str, &str)] = &[
    // ── Wave 5 W5-05: Tier 6 + the derivative families the parent plan missed ──
    ("int1e_iprinvr", "wave-5-W5-05"),
    ("int1e_iprinviprip", "wave-5-W5-05"),
    ("int1e_ipiprinvrip", "wave-5-W5-05"),
    ("int1e_rinvipiprip", "wave-5-W5-05"),
    ("int1e_iprip", "wave-5-W5-05"),
    ("int1e_ovlpip", "wave-5-W5-05"),
    ("int1e_kinip", "wave-5-W5-05"),
    // ── Wave 5 W5-06: X2C base families — LANDED 2026-08-22, removed from the
    //    baseline (they are no longer unsupported). ──
    // ── Phase 30: GIAO / property families (intor1.c, intor2.c, intor3.c, intor4.c) ──
    ("int1e_ggovlp", "phase-30"),
    ("int1e_ggnuc", "phase-30"),
    ("int1e_ggkin", "phase-30"),
    ("int1e_grjxp", "phase-30"),
    ("int1e_irpr", "phase-30"),
    ("int1e_irrp", "phase-30"),
    ("int1e_pnucxp", "phase-30"),
    ("int1e_prinvxp", "phase-30"),
    ("int1e_inuc_rxp", "phase-30"),
    ("int1e_inuc_rcxp", "phase-30"),
    ("int2e_p1vxp1", "phase-30"),
    ("int1e_sa01sp", "phase-30"),
    ("int1e_sprsp", "phase-30"),
    ("int1e_spsigmasp", "phase-30"),
    ("int1e_srsp", "phase-30"),
    ("int2e_cg_sa10sp1", "phase-30"),
    ("int2e_cg_sa10sp1spsp2", "phase-30"),
    ("int2e_giao_sa10sp1", "phase-30"),
    ("int2e_giao_sa10sp1spsp2", "phase-30"),
    ("int2e_g1spsp2", "phase-30"),
    ("int2e_pp1", "phase-30"),
    ("int2e_pp2", "phase-30"),
    ("int2e_pp1pp2", "phase-30"),
    ("int2e_spgsp1", "phase-30"),
    ("int2e_spgsp1spsp2", "phase-30"),
    ("int3c2e_ig1", "phase-30"),
    ("int3c2e_pvp1", "phase-30"),
    ("int3c2e_pvxp1", "phase-30"),
    ("int3c2e_spsp1", "phase-30"),
    ("int3c2e_spsp1ip2", "phase-30"),
    // ── Phase 31: Breit / Gaunt / gauge apex set ──
    ("int2e_gauge_r1_ssp1ssp2", "phase-31"),
    ("int2e_gauge_r1_ssp1sps2", "phase-31"),
    ("int2e_gauge_r1_sps1ssp2", "phase-31"),
    ("int2e_gauge_r1_sps1sps2", "phase-31"),
    ("int2e_gauge_r2_ssp1ssp2", "phase-31"),
    ("int2e_gauge_r2_ssp1sps2", "phase-31"),
    ("int2e_gauge_r2_sps1ssp2", "phase-31"),
    ("int2e_gauge_r2_sps1sps2", "phase-31"),
    ("int2e_cg_ssa10ssp2", "phase-31"),
    ("int2e_giao_ssa10ssp2", "phase-31"),
    ("int2e_gssp1ssp2", "phase-31"),
    ("int1e_spnuc", "phase-31"),
    ("int1e_spspsp", "phase-31"),
];

/// Both halves of the baseline correspondence, as report-ready JSON rows.
fn evaluate_parity_baseline(unsupported: &[String]) -> Vec<Value> {
    let unsupported_set: BTreeSet<&str> = unsupported.iter().map(String::as_str).collect();
    let baseline_set: BTreeSet<&str> = PARITY_BASELINE.iter().map(|(sym, _)| *sym).collect();
    let mut drift = Vec::new();
    for sym in unsupported_set.difference(&baseline_set) {
        drift.push(json!({
            "symbol": sym,
            "problem": "unsupported but not on the PARITY_BASELINE — a new omission",
        }));
    }
    for sym in baseline_set.difference(&unsupported_set) {
        drift.push(json!({
            "symbol": sym,
            "problem": "on the PARITY_BASELINE but no longer unsupported — remove the stale entry",
        }));
    }
    drift
}

fn collect_all_cint_exports() -> Result<BTreeSet<String>> {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../libcint-master/src");
    let mut files = Vec::new();
    collect_c_sources(&source_root, &mut files)?;
    files.sort();

    let mut symbols = BTreeSet::new();
    for path in files {
        let source = fs::read_to_string(&path)
            .with_context(|| format!("read libcint source `{}`", path.display()))?;
        for prefix in ["ALL_CINT(", "ALL_CINT1E("] {
            let mut rest = source.as_str();
            while let Some(offset) = rest.find(prefix) {
                rest = &rest[offset + prefix.len()..];
                let Some(end) = rest.find(')') else { break };
                let symbol = rest[..end].trim();
                if symbol != "NAME"
                    && !symbol.is_empty()
                    && symbol
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                {
                    symbols.insert(symbol.to_owned());
                }
                rest = &rest[end + 1..];
            }
        }
    }
    Ok(symbols)
}

fn collect_c_sources(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read libcint source directory `{}`", directory.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_c_sources(&path, files)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("c") {
            files.push(path);
        }
    }
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
    let lock_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../{COMPILED_MANIFEST_LOCK_JSON}"));
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

        // Dedicated-harness families (e.g. ECP) are oracle-covered but verified outside
        // the generic raw-eval matrix, so `collect_generated_symbols_for_profile` (the
        // representation matrix) omits them. Mirror that omission here, keyed on the
        // lock's top-level `canonical_family` (NOT `id.family`, which is "1e" for ECP),
        // so generated vs lock stay consistent and the drift gate does not false-positive.
        let canonical_family = entry
            .get("canonical_family")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if is_dedicated_oracle_family(canonical_family) {
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

/// Wave 5 W5-00: the three states a `stability = "stable"` row can be in.
///
/// Before W5-00 these were one bucket, so 45 rows that deliberately fail closed
/// were indistinguishable from rows that are implemented but unproven — which is
/// why `--check-lock` was red with no actionable signal.
#[derive(Default)]
struct OracleCoverageBuckets {
    /// Implemented (or believed implemented) but not yet proven byte-identical.
    /// This is the ONLY bucket that fails the gate.
    uncovered: Vec<String>,
    /// Declared API whose kernel returns `UnsupportedApi` by design, carrying an
    /// `unsupported_policy` naming the rejection site and the owning phase.
    fail_closed: Vec<Value>,
    /// Rows whose vendored libcint driver is an unconditional stub, so RULE 4
    /// byte-identity is unobtainable at any effort. cintx may evaluate them; it
    /// simply cannot prove them, and claiming coverage would be unfalsifiable.
    no_upstream_oracle: Vec<Value>,
}

fn check_oracle_coverage(lock_root: &Value) -> OracleCoverageBuckets {
    let mut buckets = OracleCoverageBuckets::default();
    let entries = match lock_root["entries"].as_array() {
        Some(e) => e,
        None => return buckets,
    };
    for entry in entries {
        let stability = entry.get("stability").and_then(Value::as_str).unwrap_or("");
        if stability != "stable" {
            continue;
        }
        let covered = entry
            .get("oracle_covered")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if covered {
            continue;
        }
        let sym = entry
            .get("id")
            .and_then(|id| id.get("symbol"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        match entry.get("unsupported_policy") {
            Some(policy) if !policy.is_null() => {
                let row = json!({
                    "symbol": sym,
                    "owner": policy.get("owner").and_then(Value::as_str).unwrap_or("?"),
                    "reason": policy.get("reason").and_then(Value::as_str).unwrap_or("?"),
                });
                match policy.get("policy").and_then(Value::as_str) {
                    Some("no_upstream_oracle") => buckets.no_upstream_oracle.push(row),
                    _ => buckets.fail_closed.push(row),
                }
            }
            _ => buckets.uncovered.push(sym.to_owned()),
        }
    }
    buckets.uncovered.sort();
    buckets
}

/// Wave 5 W5-00: a row that is `oracle_covered = true` must NOT also claim a
/// fail-closed policy — that combination means the manifest contradicts itself.
fn check_policy_contradictions(lock_root: &Value) -> Vec<String> {
    let entries = match lock_root["entries"].as_array() {
        Some(e) => e,
        None => return Vec::new(),
    };
    let mut bad = Vec::new();
    for entry in entries {
        let covered = entry
            .get("oracle_covered")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let has_policy = entry
            .get("unsupported_policy")
            .map(|p| !p.is_null())
            .unwrap_or(false);
        if covered && has_policy {
            let sym = entry
                .get("id")
                .and_then(|id| id.get("symbol"))
                .and_then(Value::as_str)
                .unwrap_or("?");
            bad.push(sym.to_owned());
        }
    }
    bad.sort();
    bad
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
                .filter(|entry| {
                    matches!(
                        entry.helper_kind,
                        HelperKind::Operator | HelperKind::SourceOnly
                    )
                })
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
