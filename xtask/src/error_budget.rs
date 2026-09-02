use anyhow::{Context, Result, bail};
use cintx_oracle::compare::generate_profile_parity_report;
use cintx_oracle::fixtures::OracleRawInputs;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const ALL_PROFILES: &[&str] = &[
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e",
    "unstable-source",
];

/// The reviewed, checked-in error envelope.  CI compares a fresh measurement
/// against this file; it is only rewritten by an explicit `--record` run.
pub const DEFAULT_BASELINE_PATH: &str = "artifacts/cintx_precision_budget.json";

#[derive(Clone, Debug)]
pub struct BudgetEntry {
    pub symbol: String,
    pub family: String,
    pub representation: String,
    pub backend: String,
    pub nroots_class: String,
    pub n_elements: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub applied_atol: f64,
    pub applied_rtol: f64,
    pub headroom_abs: f64,
    pub headroom_rel: f64,
    pub status: String,
}

impl BudgetEntry {
    fn key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.family, self.symbol, self.representation, self.backend, self.nroots_class
        )
    }

    pub fn to_json(&self) -> Value {
        json!({
            "symbol": self.symbol,
            "family": self.family,
            "representation": self.representation,
            "backend": self.backend,
            "nroots_class": self.nroots_class,
            "n_elements": self.n_elements,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "applied_atol": self.applied_atol,
            "applied_rtol": self.applied_rtol,
            "headroom": {
                "abs": self.headroom_abs,
                "rel": self.headroom_rel,
            },
            "status": self.status,
        })
    }
}

pub fn run_error_budget(
    profiles: &[String],
    check_headroom: bool,
    perturb_test: bool,
    record_baseline: bool,
    baseline_path: &Path,
) -> Result<()> {
    if check_headroom && record_baseline {
        bail!("--check-headroom and --record cannot be used together");
    }
    let active_profiles: Vec<&str> = if profiles.is_empty() {
        ALL_PROFILES.to_vec()
    } else {
        profiles.iter().map(|s| s.as_str()).collect()
    };

    // Three fixture sets, not one (`def2_speed_precision_plan.md` D4).
    //
    // `sample` carries `l = 0, 1, 0, 1`, so its 2e fixtures sit at Rys order 2
    // and the budget was blind exactly where def2-TZVP is new. The def2-shaped
    // samplers add orders 6 and 7 — one at `x = 0` on a same-centre pair and
    // one in the large-x asymptotic arm — and the third crosses those with
    // `env[PTR_RANGE_OMEGA]`, which is the newest feature against the newest.
    //
    // They are added rather than substituted: widening `sample` would move every
    // number in the checked-in envelope at once, which is a ratchet reset
    // dressed as a fixture change. Each sampler files its entries under its own
    // `nroots_class` tag, so the keys cannot collide.
    //
    // The def2 sets need the extended device Rys path to be *in the binary*:
    // their whole point is the `nroots` 6-7 classes, which without the feature
    // are refused — correctly, fail-closed — rather than measured. A build
    // without it drops them and says so, instead of reporting a sweep of
    // refusals as a precision failure.
    let mut samplers = vec![OracleRawInputs::sample()];
    if cintx_cubecl::EXTENDED_DEVICE_RYS_COMPILED {
        samplers.push(OracleRawInputs::def2_high_order());
        samplers.push(OracleRawInputs::def2_high_order_range_separated());
    } else {
        println!(
            "  NOTE: `extended-device-rys` is not compiled in, so the def2 high-order \
             fixture sets are skipped — their nroots 6-7 classes would be refused, not \
             measured. Build with --features extended-device-rys to include them."
        );
    }
    let mut entries_by_key: BTreeMap<String, BudgetEntry> = BTreeMap::new();

    for (inputs, &profile) in samplers
        .iter()
        .flat_map(|inputs| active_profiles.iter().map(move |p| (inputs, p)))
    {
        println!(
            "Evaluating precision error budget for profile `{profile}` on fixture set `{}`...",
            inputs.tag()
        );
        let include_unstable = profile == "unstable-source";
        let report = generate_profile_parity_report(inputs, profile, include_unstable)
            .with_context(|| {
                format!(
                    "generating parity report for profile `{profile}` on fixture set `{}`",
                    inputs.tag()
                )
            })?;

        for f in report.fixtures {
            let mut max_abs = f.raw_vs_upstream.max_abs_error;
            let mut max_rel = f.raw_vs_upstream.max_rel_error;

            if perturb_test && f.symbol.contains("ovlp") {
                // Phase-2 proof: remain inside the raw 1e-12 comparison while
                // exceeding a byte-identical recorded baseline. This exercises
                // the ratchet rather than merely re-proving the raw assertion.
                max_abs += 1e-13;
                max_rel += 1e-13;
            }

            let tol = f.tolerance;
            let headroom_abs = if max_abs > 0.0 {
                tol.atol / max_abs
            } else {
                f64::INFINITY
            };
            let headroom_rel = if max_rel > 0.0 {
                tol.rtol / max_rel
            } else {
                f64::INFINITY
            };

            let status = if f.skipped {
                "skipped".to_string()
            } else if max_abs <= tol.atol && max_rel <= tol.rtol {
                "pass".to_string()
            } else {
                "mismatch".to_string()
            };

            let entry = BudgetEntry {
                symbol: f.symbol.clone(),
                family: f.family.clone(),
                representation: f.representation.clone(),
                backend: f.backend.clone(),
                nroots_class: f.nroots_class.clone(),
                n_elements: f.n_elements,
                max_abs_error: max_abs,
                max_rel_error: max_rel,
                applied_atol: tol.atol,
                applied_rtol: tol.rtol,
                headroom_abs,
                headroom_rel,
                status,
            };

            // A fixture can occur in more than one measurement profile. Preserve the
            // independently worst absolute and relative error so a regression in
            // either dimension cannot be hidden by a smaller value in the other.
            entries_by_key
                .entry(entry.key())
                .and_modify(|e| {
                    e.max_abs_error = e.max_abs_error.max(entry.max_abs_error);
                    e.max_rel_error = e.max_rel_error.max(entry.max_rel_error);
                    e.headroom_abs = if e.max_abs_error > 0.0 {
                        e.applied_atol / e.max_abs_error
                    } else {
                        f64::INFINITY
                    };
                    e.headroom_rel = if e.max_rel_error > 0.0 {
                        e.applied_rtol / e.max_rel_error
                    } else {
                        f64::INFINITY
                    };
                    if entry.status == "mismatch" {
                        e.status = "mismatch".to_string();
                    }
                })
                .or_insert(entry);
        }
    }

    let all_entries: Vec<BudgetEntry> = entries_by_key.into_values().collect();

    // Summary metrics
    let evaluated_entries: Vec<&BudgetEntry> = all_entries
        .iter()
        .filter(|e| e.status != "skipped")
        .collect();

    let worst_abs_error = evaluated_entries
        .iter()
        .map(|e| e.max_abs_error)
        .fold(0.0_f64, f64::max);
    let worst_rel_error = evaluated_entries
        .iter()
        .map(|e| e.max_rel_error)
        .fold(0.0_f64, f64::max);

    let min_headroom_abs = evaluated_entries
        .iter()
        .map(|e| e.headroom_abs)
        .fold(f64::INFINITY, f64::min);
    let min_headroom_rel = evaluated_entries
        .iter()
        .map(|e| e.headroom_rel)
        .fold(f64::INFINITY, f64::min);

    let min_hr_abs_val = if min_headroom_abs.is_infinite() {
        "inf".to_string()
    } else {
        format!("{min_headroom_abs:.2e}")
    };
    let min_hr_rel_val = if min_headroom_rel.is_infinite() {
        "inf".to_string()
    } else {
        format!("{min_headroom_rel:.2e}")
    };

    let entries_json: Vec<Value> = all_entries.iter().map(BudgetEntry::to_json).collect();

    let budget_json = json!({
        "timestamp": "2026-08-30",
        "tolerance_policy": "unified_1e-12",
        "measurement": {
            "profiles": active_profiles,
            "fixture_sets": samplers.iter().map(|i| i.tag()).collect::<Vec<_>>(),
            "perturbation": perturb_test,
            // What the error columns actually compare. Not the vendor: `raw` is
            // `cintx_compat::raw::eval_raw` and `upstream` is
            // `cintx_compat::legacy`'s `cint*` wrapper for the same symbol —
            // two cintx entry points onto the same kernel. Every entry is
            // therefore 0.0 with infinite headroom, and has been since the
            // budget was introduced.
            //
            // That makes this a *path-equivalence* envelope, which is worth
            // having — the raw and legacy surfaces must not drift — but it is
            // not the cintx-vs-libcint envelope the name suggests, and
            // `--check-headroom` cannot fire on it. The vendor comparison lives
            // in `verify_legacy_wrapper_parity` (pass/fail at a flat 1e-12) and
            // in the per-family oracle parity gates, which do record measured
            // divergences.
            "comparison": "raw_api_vs_legacy_wrapper",
            "comparison_is_vendor": false,
            "vendor_envelope_source": [
                "cintx_oracle::compare::verify_legacy_wrapper_parity (flat atol=1e-12, pass/fail)",
                "crates/cintx-oracle/tests/ext_rys_*_parity.rs (extended Rys orders, vs vendored libcint 6.1.3)",
                "crates/cintx-oracle/tests/rys_ext_inline_parity.rs (inline entry vs host dispatch, bit-identity)",
            ],
        },
        "summary": {
            "total_unique_entries": all_entries.len(),
            "evaluated_entries": evaluated_entries.len(),
            "passed": evaluated_entries.iter().filter(|e| e.status == "pass").count(),
            "mismatched": evaluated_entries.iter().filter(|e| e.status == "mismatch").count(),
            "worst_max_abs_error": worst_abs_error,
            "worst_max_rel_error": worst_rel_error,
            "min_headroom_abs": min_hr_abs_val,
            "min_headroom_rel": min_hr_rel_val,
        },
        "entries": entries_json,
    });

    let json_pretty = serde_json::to_string_pretty(&budget_json)?;

    let mut failures = Vec::new();
    if evaluated_entries.iter().any(|e| e.status == "mismatch") {
        failures.push("the fresh measurement contains raw parity mismatches".to_string());
    }

    if check_headroom {
        let baseline = load_baseline(baseline_path)?;
        // Only entries this build *produced* are checked. A recorded entry the
        // run did not reach is not a regression — most often it is a def2
        // high-order row in a build without `extended-device-rys` — and
        // treating it as one would make the ratchet fire on a feature flag.
        for entry in &evaluated_entries {
            let Some(recorded) = baseline.get(&entry.key()) else {
                failures.push(format!("missing recorded budget entry for {}", entry.key()));
                continue;
            };
            if entry.max_abs_error > recorded.max_abs_error
                || entry.max_rel_error > recorded.max_rel_error
            {
                failures.push(format!(
                    "{} grew: abs {:.3e} > {:.3e} or rel {:.3e} > {:.3e}",
                    entry.key(),
                    entry.max_abs_error,
                    recorded.max_abs_error,
                    entry.max_rel_error,
                    recorded.max_rel_error,
                ));
            }
        }
    }

    // A fresh snapshot is always exported to the mandatory artifact locations.
    // The checked-in envelope is a separate, reviewed ratchet.
    let json_destinations = [
        "/mnt/data/cintx_precision_budget.current.json",
        "/tmp/cintx_artifacts/cintx_precision_budget.current.json",
        "artifacts/cintx_precision_budget_current.json",
    ];
    for dest in json_destinations {
        let p = Path::new(dest);
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(()) = fs::write(p, &json_pretty) {
            println!("  Emitted JSON error budget to: {dest}");
        }
    }

    if record_baseline {
        if perturb_test {
            bail!("refusing to record a deliberately perturbed error budget");
        }
        fs::write(baseline_path, &json_pretty).with_context(|| {
            format!(
                "writing reviewed precision budget baseline to {}",
                baseline_path.display()
            )
        })?;
        println!(
            "  Recorded reviewed headroom baseline to: {}",
            baseline_path.display()
        );
    }

    // Write human-readable Markdown report
    let date_str = "2026-08-30";
    let mut md = String::new();
    md.push_str(&format!("# Precision Error Budget Report ({date_str})\n\n"));
    md.push_str("## What the error columns compare\n\n");
    md.push_str(
        "`max_abs_error` and `max_rel_error` are **`eval_raw` against the `cint*` legacy \
         wrapper for the same symbol** — two cintx entry points onto one kernel — not cintx \
         against vendored libcint. They are consequently 0.0 with infinite headroom \
         throughout, and `--check-headroom` cannot fire on them. Read this table as a \
         path-equivalence envelope: it says the raw and legacy surfaces have not drifted \
         apart, which is a real property and not the one the word *precision* suggests.\n\n\
         The cintx-vs-libcint envelope is measured by `verify_legacy_wrapper_parity` (flat \
         `atol = 1e-12`, pass/fail) and by the per-family oracle parity gates, including \
         `ext_rys_*_parity` for the extended Rys orders.\n\n",
    );
    md.push_str("## Executive Summary\n\n");
    md.push_str(&format!(
        "- **Tolerance Model**: Unified `atol = 1.0e-12`, `rtol = 1.0e-12` across all families.\n\
         - **Evaluated Operations**: {} unique fixture combinations\n\
         - **Passed**: {}\n\
         - **Mismatches**: {}\n\
         - **Worst Observed Abs Error**: `{worst_abs_error:.3e}`\n\
         - **Worst Observed Rel Error**: `{worst_rel_error:.3e}`\n\
         - **Min Headroom (Abs)**: `{min_headroom_abs:.2e}`×\n\
         - **Min Headroom (Rel)**: `{min_headroom_rel:.2e}`×\n\n",
        evaluated_entries.len(),
        evaluated_entries
            .iter()
            .filter(|e| e.status == "pass")
            .count(),
        evaluated_entries
            .iter()
            .filter(|e| e.status == "mismatch")
            .count()
    ));

    md.push_str("## Per-Family Precision Error Budget\n\n");
    md.push_str("| Family | Symbol | Form | Backend | Nroots Class | N Elem | Max Abs Err | Max Rel Err | Headroom (Abs) | Headroom (Rel) | Status |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");

    for e in &all_entries {
        let hr_abs_str = if e.headroom_abs.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.1e}×", e.headroom_abs)
        };
        let hr_rel_str = if e.headroom_rel.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.1e}×", e.headroom_rel)
        };
        md.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | `{:.2e}` | `{:.2e}` | {} | {} | **{}** |\n",
            e.family,
            e.symbol,
            e.representation,
            e.backend,
            e.nroots_class,
            e.n_elements,
            e.max_abs_error,
            e.max_rel_error,
            hr_abs_str,
            hr_rel_str,
            e.status
        ));
    }

    let md_destinations = [
        format!("artifacts/precision_error_budget_{date_str}.md"),
        format!("/mnt/data/precision_error_budget_{date_str}.md"),
        format!("/tmp/cintx_artifacts/precision_error_budget_{date_str}.md"),
    ];
    for dest in md_destinations {
        let p = Path::new(&dest);
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(()) = fs::write(p, &md) {
            println!("  Emitted Markdown error budget to: {dest}");
        }
    }

    if !failures.is_empty() {
        bail!(
            "precision error-budget gate FAILED ({} issue(s)): {}",
            failures.len(),
            failures.join("; ")
        );
    }
    if check_headroom {
        println!("  PASS: Headroom regression check (no recorded error budget grew)");
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct RecordedBudget {
    max_abs_error: f64,
    max_rel_error: f64,
}

fn load_baseline(path: &Path) -> Result<BTreeMap<String, RecordedBudget>> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("reading precision budget baseline {}", path.display()))?;
    let json: Value = serde_json::from_str(&source)
        .with_context(|| format!("parsing precision budget baseline {}", path.display()))?;
    let entries = json
        .get("entries")
        .and_then(Value::as_array)
        .context("precision budget baseline has no entries array")?;

    let mut result = BTreeMap::new();
    for entry in entries {
        let field = |name: &str| {
            entry
                .get(name)
                .and_then(Value::as_str)
                .with_context(|| format!("baseline entry missing string field `{name}`"))
        };
        let key = format!(
            "{}:{}:{}:{}:{}",
            field("family")?,
            field("symbol")?,
            field("representation")?,
            field("backend")?,
            field("nroots_class")?,
        );
        let max_abs_error = entry
            .get("max_abs_error")
            .and_then(Value::as_f64)
            .with_context(|| format!("baseline entry {key} has no numeric max_abs_error"))?;
        let max_rel_error = entry
            .get("max_rel_error")
            .and_then(Value::as_f64)
            .with_context(|| format!("baseline entry {key} has no numeric max_rel_error"))?;
        if result
            .insert(
                key.clone(),
                RecordedBudget {
                    max_abs_error,
                    max_rel_error,
                },
            )
            .is_some()
        {
            bail!("precision budget baseline contains duplicate entry {key}");
        }
    }
    Ok(result)
}
