use anyhow::{bail, Context, Result};
use cintx_oracle::compare::{generate_profile_parity_report, verify_helper_surface_coverage};
use cintx_oracle::fixtures::{
    MATRIX_ARTIFACT_FALLBACK_NAME, OracleRawInputs, PHASE4_APPROVED_PROFILES,
    REPORT_ARTIFACT_FALLBACK_NAME, REQUIRED_MATRIX_ARTIFACT, REQUIRED_REPORT_ARTIFACT,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const REQUIRED_PROFILE_CSV: &str = "base,with-f12,with-4c1e,with-f12+with-4c1e";

/// All profiles the xtask tooling recognises (standard 4 + unstable-source nightly).
const ALL_KNOWN_PROFILES: &[&str] = &[
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e",
    "unstable-source",
];
const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const FALLBACK_ARTIFACT_DIR_DEFAULT: &str = "/tmp/cintx_artifacts";

const ORACLE_SUMMARY_REQUIRED_PATH: &str = "/mnt/data/cintx_phase_04_oracle_compare_summary.json";
const ORACLE_SUMMARY_FALLBACK_NAME: &str = "cintx_phase_04_oracle_compare_summary.json";
const HELPER_SUMMARY_REQUIRED_PATH: &str = "/mnt/data/cintx_phase_04_helper_legacy_parity.json";
const HELPER_SUMMARY_FALLBACK_NAME: &str = "cintx_phase_04_helper_legacy_parity.json";
const OOM_SUMMARY_REQUIRED_PATH: &str = "/mnt/data/cintx_phase_04_oom_contract_check.json";
const OOM_SUMMARY_FALLBACK_NAME: &str = "cintx_phase_04_oom_contract_check.json";

pub fn run_oracle_compare(profiles: &[String], include_unstable_source: bool) -> Result<()> {
    let ordered_profiles = validate_required_profile_scope(profiles)?;
    let inputs = OracleRawInputs::sample();

    let mut profile_reports = Vec::new();
    let mut profile_failures = Vec::new();
    for profile in ordered_profiles {
        let compare_result = generate_profile_parity_report(&inputs, &profile, include_unstable_source);
        let profile_slug = profile_slug(&profile);

        let matrix_source = source_path_from_result(
            compare_result.as_ref().ok().map(|report| report.matrix_artifact.actual_path.as_path()),
            REQUIRED_MATRIX_ARTIFACT,
            MATRIX_ARTIFACT_FALLBACK_NAME,
        )
        .context("resolve matrix artifact source path")?;
        let parity_source = source_path_from_result(
            compare_result.as_ref().ok().map(|report| report.parity_artifact.actual_path.as_path()),
            REQUIRED_REPORT_ARTIFACT,
            REPORT_ARTIFACT_FALLBACK_NAME,
        )
        .context("resolve parity artifact source path")?;

        let matrix_required = format!("/mnt/data/cintx_phase_04_oracle_matrix_{profile_slug}.json");
        let matrix_fallback = format!("cintx_phase_04_oracle_matrix_{profile_slug}.json");
        let parity_required = format!("/mnt/data/cintx_phase_04_oracle_compare_{profile_slug}.json");
        let parity_fallback = format!("cintx_phase_04_oracle_compare_{profile_slug}.json");

        let matrix_persisted =
            copy_artifact_with_fallback(&matrix_source, &matrix_required, &matrix_fallback)
                .with_context(|| format!("persist matrix artifact for profile `{profile}`"))?;
        let parity_persisted =
            copy_artifact_with_fallback(&parity_source, &parity_required, &parity_fallback)
                .with_context(|| format!("persist parity artifact for profile `{profile}`"))?;

        match compare_result {
            Ok(report) => {
                profile_reports.push(json!({
                    "profile": profile,
                    "status": "pass",
                    "fixture_count": report.fixture_count,
                    "mismatch_count": report.mismatch_count,
                    "include_unstable_source": include_unstable_source,
                    "matrix_artifact": matrix_persisted.to_json(),
                    "parity_artifact": parity_persisted.to_json(),
                }));
            }
            Err(error) => {
                let error_text = error.to_string();
                profile_reports.push(json!({
                    "profile": profile,
                    "status": "failed",
                    "error": error_text,
                    "include_unstable_source": include_unstable_source,
                    "matrix_artifact": matrix_persisted.to_json(),
                    "parity_artifact": parity_persisted.to_json(),
                }));
                profile_failures.push(format!("{profile}: {error}"));
            }
        }
    }

    let mut summary = json!({
        "profiles": profile_reports,
        "required_profiles_csv": REQUIRED_PROFILE_CSV,
        "include_unstable_source": include_unstable_source,
        "compiled_manifest_lock": "crates/cintx-ops/generated/compiled_manifest.lock.json",
        "status": if profile_failures.is_empty() { "ok" } else { "failed" },
    });
    let write = write_json_with_fallback(
        ORACLE_SUMMARY_REQUIRED_PATH,
        ORACLE_SUMMARY_FALLBACK_NAME,
        &summary,
    )?;
    summary["artifact_write"] = write.to_json();
    rewrite_json(&write.actual_path, &summary)?;

    if !profile_failures.is_empty() {
        bail!(
            "oracle parity gate failed for {} profile(s): {}",
            profile_failures.len(),
            profile_failures.join(" | ")
        );
    }

    println!("oracle compare summary: {}", write.actual_path.display());
    Ok(())
}

pub fn run_helper_legacy_parity(profile: &str) -> Result<()> {
    ensure_known_profile(profile)?;
    let inputs = OracleRawInputs::sample();
    verify_helper_surface_coverage(&inputs)
        .with_context(|| format!("helper/legacy parity check failed for profile `{profile}`"))?;

    let mut summary = json!({
        "profile": profile,
        "required_profiles_csv": REQUIRED_PROFILE_CSV,
        "status": "ok",
        "gate": "helper-legacy-parity",
    });
    let write = write_json_with_fallback(
        HELPER_SUMMARY_REQUIRED_PATH,
        HELPER_SUMMARY_FALLBACK_NAME,
        &summary,
    )?;
    summary["artifact_write"] = write.to_json();
    rewrite_json(&write.actual_path, &summary)?;

    println!("helper parity summary: {}", write.actual_path.display());
    Ok(())
}

pub fn run_oom_contract_check() -> Result<()> {
    let commands: Vec<Vec<&str>> = vec![
        vec![
            "test",
            "-p",
            "cintx-compat",
            "raw::tests::memory_limit_failure_keeps_output_slice_unchanged",
            "--",
            "--exact",
        ],
        vec![
            "test",
            "-p",
            "cintx-runtime",
            "workspace::tests::chunk_planner_reports_limit_exceeded_when_no_chunk_can_fit",
            "--",
            "--exact",
        ],
    ];

    for args in &commands {
        run_cargo_command(args)?;
    }

    let mut summary = json!({
        "status": "ok",
        "gate": "oom-contract-check",
        "executed_commands": commands,
    });
    let write = write_json_with_fallback(OOM_SUMMARY_REQUIRED_PATH, OOM_SUMMARY_FALLBACK_NAME, &summary)?;
    summary["artifact_write"] = write.to_json();
    rewrite_json(&write.actual_path, &summary)?;

    println!("oom contract summary: {}", write.actual_path.display());
    Ok(())
}

fn run_cargo_command(args: &[&str]) -> Result<()> {
    let status = Command::new("cargo")
        .args(args)
        .status()
        .with_context(|| format!("spawn cargo {}", args.join(" ")))?;
    if !status.success() {
        bail!("cargo {} failed with status {status}", args.join(" "));
    }
    Ok(())
}

fn validate_required_profile_scope(profiles: &[String]) -> Result<Vec<String>> {
    // Validate all requested profiles are known
    for p in profiles {
        if !ALL_KNOWN_PROFILES.contains(&p.as_str()) {
            bail!(
                "unknown profile `{p}`, expected one of: {}",
                ALL_KNOWN_PROFILES.join(", ")
            );
        }
    }
    // If requesting unstable-source, it runs standalone (per D-02)
    if profiles.iter().any(|p| p == "unstable-source") {
        if profiles.len() != 1 || profiles[0] != "unstable-source" {
            bail!("unstable-source profile must be run standalone, not combined with other profiles");
        }
        return Ok(vec!["unstable-source".to_owned()]);
    }
    // Standard 4-profile validation
    let requested: BTreeSet<String> = profiles.iter().cloned().collect();
    let required: BTreeSet<String> = PHASE4_APPROVED_PROFILES
        .iter()
        .map(|profile| (*profile).to_owned())
        .collect();
    let missing: Vec<String> = required.difference(&requested).cloned().collect();
    let extra: Vec<String> = requested.difference(&required).cloned().collect();
    if !missing.is_empty() || !extra.is_empty() {
        bail!(
            "profile scope mismatch, expected exactly `{REQUIRED_PROFILE_CSV}` (missing: {:?}, extra: {:?})",
            missing,
            extra
        );
    }
    Ok(PHASE4_APPROVED_PROFILES
        .iter()
        .map(|profile| (*profile).to_owned())
        .collect())
}

fn ensure_known_profile(profile: &str) -> Result<()> {
    if ALL_KNOWN_PROFILES.contains(&profile) {
        return Ok(());
    }
    bail!(
        "unsupported profile `{profile}`, expected one of: {}",
        ALL_KNOWN_PROFILES.join(", ")
    )
}

fn profile_slug(profile: &str) -> String {
    profile
        .replace('+', "_plus_")
        .replace('-', "_")
}

fn source_path_from_result(
    from_report: Option<&Path>,
    required_path: &str,
    fallback_name: &str,
) -> Result<PathBuf> {
    if let Some(path) = from_report {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
    }

    let required = PathBuf::from(required_path);
    if required.is_file() {
        return Ok(required);
    }
    let fallback = fallback_dir().join(fallback_name);
    if fallback.is_file() {
        return Ok(fallback);
    }
    bail!(
        "artifact source missing (required: `{required_path}`, fallback: `{}`)",
        fallback.display()
    );
}

fn copy_artifact_with_fallback(
    source: &Path,
    required_target: &str,
    fallback_name: &str,
) -> Result<ArtifactWrite> {
    let payload = fs::read(source).with_context(|| format!("read source artifact `{}`", source.display()))?;
    write_bytes_with_fallback(required_target, fallback_name, &payload)
}

fn write_json_with_fallback(
    required_path: &str,
    fallback_name: &str,
    value: &Value,
) -> Result<ArtifactWrite> {
    let payload = serde_json::to_vec_pretty(value).context("serialize json artifact")?;
    write_bytes_with_fallback(required_path, fallback_name, &payload)
}

fn write_bytes_with_fallback(
    required_path: &str,
    fallback_name: &str,
    payload: &[u8],
) -> Result<ArtifactWrite> {
    let required = PathBuf::from(required_path);
    match try_write_payload(&required, payload) {
        Ok(()) => Ok(ArtifactWrite {
            required_path: required_path.to_owned(),
            actual_path: required,
            used_required_path: true,
            fallback_reason: None,
        }),
        Err(error) => {
            let fallback = fallback_dir().join(fallback_name);
            try_write_payload(&fallback, payload).with_context(|| {
                format!(
                    "failed to write fallback artifact `{}` after required-path failure",
                    fallback.display()
                )
            })?;
            Ok(ArtifactWrite {
                required_path: required_path.to_owned(),
                actual_path: fallback,
                used_required_path: false,
                fallback_reason: Some(error.to_string()),
            })
        }
    }
}

fn try_write_payload(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create artifact parent directory `{}`", parent.display()))?;
    }
    fs::write(path, payload).with_context(|| format!("write artifact `{}`", path.display()))?;
    Ok(())
}

fn fallback_dir() -> PathBuf {
    env::var(FALLBACK_ARTIFACT_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(FALLBACK_ARTIFACT_DIR_DEFAULT))
}

fn rewrite_json(path: &Path, value: &Value) -> Result<()> {
    let payload = serde_json::to_vec_pretty(value).context("serialize final artifact json")?;
    fs::write(path, payload).with_context(|| format!("rewrite artifact `{}`", path.display()))?;
    Ok(())
}

#[derive(Clone, Debug)]
struct ArtifactWrite {
    required_path: String,
    actual_path: PathBuf,
    used_required_path: bool,
    fallback_reason: Option<String>,
}

impl ArtifactWrite {
    fn to_json(&self) -> Value {
        json!({
            "required_path": self.required_path,
            "actual_path": self.actual_path.display().to_string(),
            "used_required_path": self.used_required_path,
            "fallback_reason": self.fallback_reason,
            "fallback_env_var": FALLBACK_ARTIFACT_DIR_ENV,
        })
    }
}
