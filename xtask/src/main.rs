mod bench_report;
mod gen_ecp_tables;
mod gen_rys_tables;
mod manifest_audit;
mod oracle_covered_update;
mod oracle_update;
mod rocm_oracle;
mod wgpu_capability_gate;

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Context, Result};

const REQUIRED_PROFILES_CSV: &str = "base,with-f12,with-4c1e,with-f12+with-4c1e";
const REQUIRED_PROFILES: [&str; 4] = ["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"];
const ORACLE_COMPARE_PROFILES_CSV: &str =
    "base,with-f12,with-4c1e,with-f12+with-4c1e,unstable-source";
const ORACLE_COMPARE_PROFILES: [&str; 5] = [
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e",
    "unstable-source",
];

#[derive(Debug)]
enum Command {
    ManifestAudit {
        profiles: Vec<String>,
        check_lock: bool,
    },
    BenchReport {
        thresholds_path: String,
        mode: String,
    },
    OracleCompare {
        profiles: Vec<String>,
        include_unstable_source: bool,
    },
    HelperLegacyParity {
        profile: String,
    },
    OomContractCheck,
    OracleCoveredUpdate,
    /// Phase 16-04 / D-15 ROCm oracle wrapper. Operator-driven — not in CI.
    RocmOracle {
        profile: Option<String>,
    },
    WgpuCapabilityGate {
        profiles: Vec<String>,
        require_adapter: bool,
    },
    /// Phase 19 D-15. Extract PySCF K-Taylor tables into `.bin` blobs; `--check`
    /// re-extracts and fails closed if the committed blob diverges from source.
    GenEcpTables {
        check: bool,
    },
    /// Phase 25 FND-02 / D-04. Extract libcint Jacobi/Flocke Rys tables into
    /// `roots_jacobi_data.rs`; `--check` re-derives and fails closed if the
    /// committed module diverges from the vendored C source.
    GenRysTables {
        check: bool,
    },
    Help,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("xtask gate failed: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command_name) = args.next() else {
        print_help();
        return Ok(());
    };

    let command = match command_name.as_str() {
        "manifest-audit" => parse_manifest_audit(args)?,
        "bench-report" => parse_bench_report(args)?,
        "oracle-compare" => parse_oracle_compare(args)?,
        "helper-legacy-parity" => parse_helper_legacy_parity(args)?,
        "oom-contract-check" => parse_oom_contract_check(args)?,
        "oracle-covered-update" => Command::OracleCoveredUpdate,
        "rocm-oracle" => parse_rocm_oracle(args)?,
        "wgpu-capability-gate" => parse_wgpu_capability_gate(args)?,
        "gen-ecp-tables" => parse_gen_ecp_tables(args)?,
        "gen-rys-tables" => parse_gen_rys_tables(args)?,
        "--help" | "-h" | "help" => Command::Help,
        other => return Err(anyhow!("unknown xtask command: {other}")),
    };

    execute(command)
}

fn execute(command: Command) -> Result<()> {
    match command {
        Command::ManifestAudit {
            profiles,
            check_lock,
        } => manifest_audit::run_manifest_audit(&profiles, check_lock),
        Command::BenchReport {
            thresholds_path,
            mode,
        } => bench_report::run_bench_report(&thresholds_path, &mode),
        Command::OracleCompare {
            profiles,
            include_unstable_source,
        } => oracle_update::run_oracle_compare(&profiles, include_unstable_source),
        Command::HelperLegacyParity { profile } => oracle_update::run_helper_legacy_parity(&profile),
        Command::OomContractCheck => oracle_update::run_oom_contract_check(),
        Command::OracleCoveredUpdate => oracle_covered_update::run_oracle_covered_update(),
        Command::RocmOracle { profile } => rocm_oracle::run_rocm_oracle(profile.as_deref()),
        Command::WgpuCapabilityGate {
            profiles,
            require_adapter,
        } => wgpu_capability_gate::run_wgpu_capability_gate(&profiles, require_adapter),
        Command::GenEcpTables { check } => gen_ecp_tables::run_gen_ecp_tables(check),
        Command::GenRysTables { check } => gen_rys_tables::run_gen_rys_tables(check),
        Command::Help => {
            print_help();
            Ok(())
        }
    }
}

fn parse_manifest_audit(args: impl Iterator<Item = String>) -> Result<Command> {
    let mut profiles = required_profiles();
    let mut check_lock = false;
    let items: Vec<String> = args.collect();
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--profiles" => {
                let csv = items
                    .get(index + 1)
                    .context("expected csv value after --profiles")?;
                profiles = parse_profiles_csv(csv)?;
                index += 2;
            }
            "--check-lock" => {
                check_lock = true;
                index += 1;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown manifest-audit flag: {other}")),
        }
    }
    Ok(Command::ManifestAudit {
        profiles,
        check_lock,
    })
}

fn parse_oracle_compare(args: impl Iterator<Item = String>) -> Result<Command> {
    let mut profiles = required_profiles();
    let mut include_unstable_source = false;
    let items: Vec<String> = args.collect();
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--profiles" => {
                let csv = items
                    .get(index + 1)
                    .context("expected csv value after --profiles")?;
                profiles = parse_oracle_compare_profiles_csv(csv)?;
                index += 2;
            }
            "--include-unstable-source" => {
                let value = items
                    .get(index + 1)
                    .context("expected boolean value after --include-unstable-source")?;
                include_unstable_source = parse_boolean(value, "--include-unstable-source")?;
                index += 2;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown oracle-compare flag: {other}")),
        }
    }
    Ok(Command::OracleCompare {
        profiles,
        include_unstable_source,
    })
}

fn parse_bench_report(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut thresholds_path = String::from("ci/benchmark-thresholds.json");
    let mut mode = String::from("enforce");
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--thresholds" => {
                let value = items
                    .get(index + 1)
                    .context("expected path after --thresholds")?;
                thresholds_path = value.clone();
                index += 2;
            }
            "--mode" => {
                let value = items
                    .get(index + 1)
                    .context("expected value after --mode")?;
                mode = value.clone();
                index += 2;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown bench-report flag: {other}")),
        }
    }
    Ok(Command::BenchReport {
        thresholds_path,
        mode,
    })
}

fn parse_helper_legacy_parity(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut profile = String::from("base");
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--profile" => {
                let value = items
                    .get(index + 1)
                    .context("expected value after --profile")?;
                ensure_known_profile(value)?;
                profile = value.clone();
                index += 2;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown helper-legacy-parity flag: {other}")),
        }
    }
    Ok(Command::HelperLegacyParity { profile })
}

fn parse_rocm_oracle(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut profile: Option<String> = None;
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--profile" => {
                let value = items
                    .get(index + 1)
                    .context("expected value after --profile")?;
                ensure_known_profile(value)?;
                profile = Some(value.clone());
                index += 2;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown rocm-oracle flag: {other}")),
        }
    }
    Ok(Command::RocmOracle { profile })
}

fn parse_oom_contract_check(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    if items.iter().any(|item| item == "--help" || item == "-h") {
        return Ok(Command::Help);
    }
    if let Some(flag) = items.first() {
        return Err(anyhow!("unknown oom-contract-check flag: {flag}"));
    }
    Ok(Command::OomContractCheck)
}

fn parse_wgpu_capability_gate(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut profiles = required_profiles();
    let mut require_adapter = false;
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--profiles" => {
                let csv = items
                    .get(index + 1)
                    .context("expected csv value after --profiles")?;
                profiles = parse_profiles_csv(csv)?;
                index += 2;
            }
            "--require-adapter" => {
                let value = items
                    .get(index + 1)
                    .context("expected boolean value after --require-adapter")?;
                require_adapter = parse_boolean(value, "--require-adapter")?;
                index += 2;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => {
                return Err(anyhow!(
                    "unknown wgpu-capability-gate flag: {other}"
                ))
            }
        }
    }
    Ok(Command::WgpuCapabilityGate {
        profiles,
        require_adapter,
    })
}

fn parse_gen_ecp_tables(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut check = false;
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--check" => {
                check = true;
                index += 1;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown gen-ecp-tables flag: {other}")),
        }
    }
    Ok(Command::GenEcpTables { check })
}

fn parse_gen_rys_tables(args: impl Iterator<Item = String>) -> Result<Command> {
    let items: Vec<String> = args.collect();
    let mut check = false;
    let mut index = 0;
    while let Some(flag) = items.get(index) {
        match flag.as_str() {
            "--check" => {
                check = true;
                index += 1;
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(anyhow!("unknown gen-rys-tables flag: {other}")),
        }
    }
    Ok(Command::GenRysTables { check })
}

fn parse_profiles_csv(csv: &str) -> Result<Vec<String>> {
    parse_profiles_csv_with_allowlist(csv, &REQUIRED_PROFILES, REQUIRED_PROFILES_CSV)
}

fn parse_oracle_compare_profiles_csv(csv: &str) -> Result<Vec<String>> {
    parse_profiles_csv_with_allowlist(
        csv,
        &ORACLE_COMPARE_PROFILES,
        ORACLE_COMPARE_PROFILES_CSV,
    )
}

fn parse_profiles_csv_with_allowlist(
    csv: &str,
    known_profiles: &[&str],
    known_profiles_csv: &str,
) -> Result<Vec<String>> {
    let values: Vec<String> = csv
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if values.is_empty() {
        bail!("profiles cannot be empty");
    }
    let unique: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    if unique.len() != values.len() {
        bail!("profiles list contains duplicates: {csv}");
    }
    for value in &values {
        ensure_known_profile_with_allowlist(value, known_profiles, known_profiles_csv)?;
    }
    Ok(values)
}

fn ensure_known_profile(profile: &str) -> Result<()> {
    ensure_known_profile_with_allowlist(profile, &REQUIRED_PROFILES, REQUIRED_PROFILES_CSV)
}

fn ensure_known_profile_with_allowlist(
    profile: &str,
    known_profiles: &[&str],
    known_profiles_csv: &str,
) -> Result<()> {
    if known_profiles.contains(&profile) {
        return Ok(());
    }
    Err(anyhow!(
        "unsupported profile '{profile}', expected one of: {known_profiles_csv}"
    ))
}

fn parse_boolean(value: &str, flag: &str) -> Result<bool> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(anyhow!(
            "invalid boolean for {flag}: {value} (expected true/false)"
        )),
    }
}

fn required_profiles() -> Vec<String> {
    REQUIRED_PROFILES_CSV
        .split(',')
        .map(ToOwned::to_owned)
        .collect()
}

fn print_help() {
    println!("xtask verification gates");
    println!();
    println!("Usage:");
    println!("  cargo run --manifest-path xtask/Cargo.toml -- <command> [options]");
    println!();
    println!("Commands:");
    println!("  manifest-audit [--profiles {REQUIRED_PROFILES_CSV}] [--check-lock]");
    println!("  bench-report [--thresholds ci/benchmark-thresholds.json] [--mode enforce|calibration]");
    println!(
        "  oracle-compare [--profiles {ORACLE_COMPARE_PROFILES_CSV}] [--include-unstable-source true|false]"
    );
    println!("  helper-legacy-parity [--profile base]");
    println!("  oom-contract-check");
    println!("  oracle-covered-update                      Run oracle parity for all 4 profiles and stamp oracle_covered=true in manifest lock");
    println!("  rocm-oracle [--profile base]               Run ROCm oracle base-family suite (env-gated; requires --features rocm and ROCm 7.x on dev host; D-15: not in CI)");
    println!("  wgpu-capability-gate [--profiles {REQUIRED_PROFILES_CSV}] [--require-adapter true|false]");
    println!("  gen-ecp-tables [--check]                   Extract PySCF K-Taylor tables to .bin blobs; --check is a byte-exact drift gate (Phase 19 D-15)");
    println!();
    println!("Defaults:");
    println!("  profiles: {REQUIRED_PROFILES_CSV}");
    println!("  include_unstable_source: false");
    println!("  require_adapter: false");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_compare_accepts_unstable_source_profile() {
        let command = parse_oracle_compare(
            vec![
                "--profiles".to_owned(),
                "unstable-source".to_owned(),
                "--include-unstable-source".to_owned(),
                "true".to_owned(),
            ]
            .into_iter(),
        )
        .expect("oracle-compare parser should accept unstable-source profile");

        match command {
            Command::OracleCompare {
                profiles,
                include_unstable_source,
            } => {
                assert_eq!(profiles, vec!["unstable-source".to_owned()]);
                assert!(include_unstable_source);
            }
            other => panic!("unexpected command variant: {other:?}"),
        }
    }

    #[test]
    fn manifest_audit_still_rejects_unstable_source_profile() {
        let error = parse_manifest_audit(
            vec!["--profiles".to_owned(), "unstable-source".to_owned()].into_iter(),
        )
        .expect_err("manifest-audit parser should reject unstable-source profile");

        let text = error.to_string();
        assert!(
            text.contains("unsupported profile 'unstable-source'"),
            "unexpected error text: {text}"
        );
        assert!(
            text.contains(REQUIRED_PROFILES_CSV),
            "error should point to required profile allowlist: {text}"
        );
    }
}
