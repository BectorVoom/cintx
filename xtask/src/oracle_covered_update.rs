use anyhow::{Context, Result};
use cintx_oracle::compare::generate_profile_parity_report;
use cintx_oracle::fixtures::{OracleRawInputs, PHASE4_APPROVED_PROFILES};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const COMPILED_MANIFEST_LOCK_PATH: &str = "crates/cintx-ops/generated/compiled_manifest.lock.json";

pub fn run_oracle_covered_update() -> Result<()> {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../{COMPILED_MANIFEST_LOCK_PATH}"));
    let lock_text = fs::read_to_string(&lock_path)
        .with_context(|| format!("read lock at {}", lock_path.display()))?;
    let mut lock: Value = serde_json::from_str(&lock_text)
        .context("parse compiled manifest lock")?;

    let inputs = OracleRawInputs::sample();

    // Collect symbols that passed oracle parity in each profile.
    // generate_profile_parity_report bails if mismatch_count > 0, so any
    // symbol present in the returned report passed parity.
    let mut covered_symbols: BTreeSet<String> = BTreeSet::new();
    for profile in PHASE4_APPROVED_PROFILES {
        println!("running oracle parity for profile: {profile}");
        let report = generate_profile_parity_report(&inputs, profile, false)
            .with_context(|| format!("oracle parity failed for profile `{profile}`"))?;

        for fixture in &report.fixtures {
            covered_symbols.insert(fixture.symbol.clone());
        }
    }

    let entries = lock
        .get_mut("entries")
        .and_then(Value::as_array_mut)
        .context("lock missing entries array")?;

    let mut stamped_count = 0usize;
    for entry in entries.iter_mut() {
        let stability = entry
            .get("stability")
            .and_then(Value::as_str)
            .unwrap_or("");
        // Only stamp stable and optional entries (not unstable_source per D-07).
        if !matches!(stability, "stable" | "optional") {
            continue;
        }

        let symbol = entry
            .get("id")
            .and_then(|id| id.get("symbol"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let helper_kind = entry
            .get("helper_kind")
            .and_then(Value::as_str)
            .unwrap_or("operator");

        // Operator and source_only symbols: covered if they appeared in parity fixtures.
        // Helper/transform/optimizer/legacy symbols: covered because
        // verify_helper_surface_coverage passes as part of build_profile_parity_report.
        let is_covered = if matches!(helper_kind, "operator" | "source_only") {
            covered_symbols.contains(symbol)
        } else {
            // helper, transform, optimizer, legacy — all verified by verify_helper_surface_coverage
            true
        };

        if is_covered {
            entry["oracle_covered"] = serde_json::json!(true);
            stamped_count += 1;
        }
    }

    let output = serde_json::to_vec_pretty(&lock).context("serialize updated lock")?;
    fs::write(&lock_path, output)
        .with_context(|| format!("write updated lock to {}", lock_path.display()))?;

    println!("oracle-covered-update: stamped {stamped_count} entries as oracle_covered=true");
    Ok(())
}
