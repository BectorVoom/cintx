use anyhow::{Context, Result, anyhow};
use cintx_compat::helpers::{CINTcgto_cart, CINTcgto_spheric, CINTcgto_spinor};
use cintx_core::Representation;
use cintx_ops::resolver::{HelperKind, ManifestEntry, Resolver};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const REQUIRED_MATRIX_ARTIFACT: &str =
    "/mnt/data/cintx_phase_02_manifest_representation_matrix.json";
pub const MATRIX_ARTIFACT_FALLBACK_NAME: &str =
    "cintx_phase_02_manifest_representation_matrix.json";
pub const REQUIRED_REPORT_ARTIFACT: &str = "/mnt/data/cintx_phase_02_compat_parity_report.json";
pub const REPORT_ARTIFACT_FALLBACK_NAME: &str = "cintx_phase_02_compat_parity_report.json";
pub const PHASE2_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e"];
pub const COMPILED_MANIFEST_LOCK_JSON: &str =
    include_str!("../../cintx-ops/generated/compiled_manifest.lock.json");
const BASE_PROFILE: &str = "base";
const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const FALLBACK_ARTIFACT_DIR_DEFAULT: &str = "/tmp/cintx_artifacts";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleFixture {
    pub family: String,
    pub symbol: String,
    pub representation: String,
    pub arity: usize,
    pub dims: Vec<usize>,
    pub component_count: usize,
    pub complex_interleaved: bool,
}

impl OracleFixture {
    pub fn required_elements(&self) -> usize {
        let base = self
            .dims
            .iter()
            .fold(self.component_count.max(1), |acc, extent| {
                acc.saturating_mul(*extent)
            });
        if self.complex_interleaved {
            base.saturating_mul(2)
        } else {
            base
        }
    }
}

#[derive(Clone, Debug)]
pub struct OracleRawInputs {
    pub atm: Vec<i32>,
    pub bas: Vec<i32>,
    pub env: Vec<f64>,
    shls2: Vec<i32>,
    shls3: Vec<i32>,
    shls4: Vec<i32>,
}

impl OracleRawInputs {
    pub fn sample() -> Self {
        // env layout:
        // 0..3 coordinates, then (exp, coeff) scalar pairs for 4 shells.
        let env = vec![
            0.0, 0.0, 0.0, // coord
            1.0, 1.0, // shell 0
            0.9, 0.8, // shell 1
            0.7, 0.6, // shell 2
            0.5, 0.4, // shell 3
        ];
        let atm = vec![
            1, // charge / atomic number
            0, // PTR_COORD
            1, // point charge model
            0, // PTR_ZETA
            0, // PTR_FRAC_CHARGE
            0,
        ];
        let bas = vec![
            0, 0, 1, 1, 0, 3, 4, 0, // shell 0 (s)
            0, 1, 1, 1, 0, 5, 6, 0, // shell 1 (p)
            0, 0, 1, 1, 0, 7, 8, 0, // shell 2 (s)
            0, 1, 1, 1, 0, 9, 10, 0, // shell 3 (p)
        ];

        Self {
            atm,
            bas,
            env,
            shls2: vec![0, 1],
            shls3: vec![0, 1, 2],
            shls4: vec![0, 1, 2, 3],
        }
    }

    pub fn shells_for_arity(&self, arity: usize) -> &[i32] {
        match arity {
            2 => &self.shls2,
            3 => &self.shls3,
            4 => &self.shls4,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactWriteResult {
    pub required_path: &'static str,
    pub actual_path: PathBuf,
    pub used_required_path: bool,
    pub fallback_reason: Option<String>,
}

pub fn write_pretty_json_artifact(
    required_path: &'static str,
    fallback_name: &str,
    value: &Value,
) -> Result<ArtifactWriteResult> {
    let payload = serde_json::to_vec_pretty(value).context("serialize artifact json")?;
    let required = Path::new(required_path);
    match try_write_payload(required, &payload) {
        Ok(()) => Ok(ArtifactWriteResult {
            required_path,
            actual_path: required.to_path_buf(),
            used_required_path: true,
            fallback_reason: None,
        }),
        Err(error) => {
            let fallback_dir = std::env::var(FALLBACK_ARTIFACT_DIR_ENV)
                .unwrap_or_else(|_| FALLBACK_ARTIFACT_DIR_DEFAULT.to_owned());
            let fallback = Path::new(&fallback_dir).join(fallback_name);
            try_write_payload(&fallback, &payload).with_context(|| {
                format!(
                    "failed to write fallback artifact `{}` after required-path failure",
                    fallback.display()
                )
            })?;
            Ok(ArtifactWriteResult {
                required_path,
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

fn representation_from_entry(entry: &ManifestEntry) -> Option<Representation> {
    match (
        entry.representation.cart,
        entry.representation.spheric,
        entry.representation.spinor,
    ) {
        (true, false, false) => Some(Representation::Cart),
        (false, true, false) => Some(Representation::Spheric),
        (false, false, true) => Some(Representation::Spinor),
        _ => None,
    }
}

fn representation_name(representation: Representation) -> &'static str {
    match representation {
        Representation::Cart => "cart",
        Representation::Spheric => "spheric",
        Representation::Spinor => "spinor",
    }
}

fn parse_component_count(component_rank: &str) -> Result<usize> {
    let trimmed = component_rank.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("scalar") {
        return Ok(1);
    }

    let mut count = 1usize;
    let mut found = false;
    for segment in trimmed.split(|ch: char| !ch.is_ascii_digit()) {
        if segment.is_empty() {
            continue;
        }
        let value = segment.parse::<usize>().with_context(|| {
            format!("parse component segment `{segment}` from `{component_rank}`")
        })?;
        count = count
            .checked_mul(value)
            .ok_or_else(|| anyhow!("component rank overflow for `{component_rank}`"))?;
        found = true;
    }
    if !found {
        return Ok(1);
    }
    Ok(count)
}

fn ao_count_for_rep(shell: i32, representation: Representation, bas: &[i32]) -> Result<usize> {
    match representation {
        Representation::Cart => {
            CINTcgto_cart(shell, bas).with_context(|| format!("cart ao count for shell {shell}"))
        }
        Representation::Spheric => CINTcgto_spheric(shell, bas)
            .with_context(|| format!("spheric ao count for shell {shell}")),
        Representation::Spinor => CINTcgto_spinor(shell, bas)
            .with_context(|| format!("spinor ao count for shell {shell}")),
    }
}

fn dims_for_arity(
    inputs: &OracleRawInputs,
    representation: Representation,
    arity: usize,
) -> Result<Vec<usize>> {
    inputs
        .shells_for_arity(arity)
        .iter()
        .copied()
        .map(|shell| ao_count_for_rep(shell, representation, &inputs.bas))
        .collect()
}

fn phase2_operator_entries() -> Vec<&'static ManifestEntry> {
    let mut entries: Vec<_> = Resolver::manifest()
        .iter()
        .filter(|entry| {
            matches!(entry.helper_kind, HelperKind::Operator)
                && entry.compiled_in_profiles.contains(&BASE_PROFILE)
                && PHASE2_FAMILIES.contains(&entry.family_name)
        })
        .collect();
    entries.sort_by_key(|entry| entry.symbol_name);
    entries
}

pub fn phase2_manifest_symbols() -> BTreeSet<String> {
    phase2_operator_entries()
        .into_iter()
        .map(|entry| entry.symbol_name.to_owned())
        .collect()
}

pub fn manifest_lock_symbols() -> Result<BTreeSet<String>> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON)
        .context("parse compiled manifest lock")?;
    let entries = root
        .get("entries")
        .and_then(Value::as_array)
        .context("compiled manifest lock missing `entries` array")?;

    let mut symbols = BTreeSet::new();
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(Value::as_object)
            .context("compiled manifest entry missing `id`")?;
        let family = id.get("family").and_then(Value::as_str).unwrap_or_default();
        let Some(symbol) = id.get("symbol").and_then(Value::as_str) else {
            continue;
        };
        let Some(_representation) = id.get("representation").and_then(Value::as_str) else {
            continue;
        };
        if !PHASE2_FAMILIES.contains(&family) {
            continue;
        }
        let has_base = entry
            .get("profiles")
            .and_then(Value::as_array)
            .map(|profiles| {
                profiles
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|profile| profile == BASE_PROFILE)
            })
            .unwrap_or(false);
        if has_base {
            symbols.insert(symbol.to_owned());
        }
    }

    Ok(symbols)
}

pub fn build_phase2_representation_matrix(inputs: &OracleRawInputs) -> Result<Vec<OracleFixture>> {
    let mut fixtures = Vec::new();
    for entry in phase2_operator_entries() {
        let Some(representation) = representation_from_entry(entry) else {
            continue;
        };
        let dims = dims_for_arity(inputs, representation, usize::from(entry.arity))
            .with_context(|| format!("derive dims for `{}`", entry.symbol_name))?;
        fixtures.push(OracleFixture {
            family: entry.family_name.to_owned(),
            symbol: entry.symbol_name.to_owned(),
            representation: representation_name(representation).to_owned(),
            arity: usize::from(entry.arity),
            dims,
            component_count: parse_component_count(entry.component_rank)
                .with_context(|| format!("component_count for `{}`", entry.symbol_name))?,
            complex_interleaved: matches!(representation, Representation::Spinor),
        });
    }
    fixtures.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    Ok(fixtures)
}

pub fn write_representation_matrix_artifact(
    matrix: &[OracleFixture],
) -> Result<ArtifactWriteResult> {
    let fixtures_json: Vec<Value> = matrix
        .iter()
        .map(|fixture| {
            json!({
                "family": fixture.family,
                "symbol": fixture.symbol,
                "representation": fixture.representation,
                "arity": fixture.arity,
                "dims": fixture.dims,
                "component_count": fixture.component_count,
                "complex_interleaved": fixture.complex_interleaved,
                "required_elements": fixture.required_elements(),
            })
        })
        .collect();

    let artifact = json!({
        "representation_matrix": fixtures_json,
        "required_path": REQUIRED_MATRIX_ARTIFACT,
        "compiled_manifest": "crates/cintx-ops/generated/compiled_manifest.lock.json",
        "families": PHASE2_FAMILIES,
    });

    write_pretty_json_artifact(
        REQUIRED_MATRIX_ARTIFACT,
        MATRIX_ARTIFACT_FALLBACK_NAME,
        &artifact,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_matrix_matches_manifest_fixtures() {
        let inputs = OracleRawInputs::sample();
        let matrix = build_phase2_representation_matrix(&inputs).expect("matrix");
        let actual: BTreeSet<String> = matrix
            .iter()
            .map(|fixture| fixture.symbol.clone())
            .collect();

        let expected = phase2_manifest_symbols();
        assert_eq!(actual, expected);

        let lock = manifest_lock_symbols().expect("lock symbols");
        assert_eq!(actual, lock);
    }

    #[test]
    fn representation_matrix_artifact_is_written() {
        let inputs = OracleRawInputs::sample();
        let matrix = build_phase2_representation_matrix(&inputs).expect("matrix");
        let written = write_representation_matrix_artifact(&matrix).expect("artifact write");
        assert!(
            written.actual_path.is_file(),
            "artifact must exist at `{}`",
            written.actual_path.display()
        );

        let content = fs::read_to_string(&written.actual_path).expect("artifact content");
        assert!(content.contains("representation_matrix"));
        assert!(content.contains(REQUIRED_MATRIX_ARTIFACT));
    }
}
