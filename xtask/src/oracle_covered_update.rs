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
            // Skipped fixtures carry no numeric parity obligation. They are recorded as
            // passing to keep fixture_count == fixtures.len(), but must NOT be stamped
            // oracle_covered=true — doing so would be a false verification claim
            // (threat T-21-08-02).
            //
            // Phase 27 (D-12, 2026-05-31): the sf-derivative SPINOR families are now
            // oracle_covered via real libcint 6.1.3 vendor byte-identity parity (Plans
            // 03/04, see crates/cintx-oracle/tests/spinor_deriv_parity.rs) — the 18
            // arity-2 1e ip families (sf_2d) plus int3c2e_ip1/ip2_spinor (sf_3c2e).
            // EXCEPTION — four vendor-stub arms stay oracle_covered=false because
            // libcint 6.1.3 ships them as unimplemented stubs, so NO byte-identity
            // reference is achievable:
            //   - int2c2e_ip1_spinor / int2c2e_ip2_spinor  -> stub `return 0` (all-zero)
            //   - int3c1e_ip1_spinor / int3c1e_iprinv_spinor -> CINT3c1e_spinor_drv exit(1)
            // Their parity tests stay #[ignore]'d; finite-difference verification is a
            // deferred follow-up. The D-03 arity-4 int2e_ip* spinor families and the
            // D-04 int1e_ecp_iprinv_spinor family also remain intentionally
            // skipped/deferred. The `skipped` guard below is what keeps every such
            // deferred family from ever being stamped covered.
            if fixture.skipped {
                continue;
            }
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
