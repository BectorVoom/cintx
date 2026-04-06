use anyhow::{anyhow, bail, Context, Result};
use cintx_compat::helpers::{CINTcgto_cart, CINTcgto_spheric, CINTcgto_spinor};
use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_F12_ZETA, PTR_ZETA,
};
use cintx_core::Representation;
use cintx_ops::resolver::{HelperKind, ManifestEntry, Resolver};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G molecular fixture (PTR_ENV_START-aligned)
// ─────────────────────────────────────────────────────────────────────────────

/// Build H2O STO-3G libcint-style atm/bas/env with user data starting at PTR_ENV_START.
///
/// PTR_ENV_START alignment is required for 2e-family integrals (2c2e, 3c2e, 2e)
/// to avoid corrupting libcint global env slots (e.g., PTR_RANGE_OMEGA at index 8).
///
/// Molecule: H2O with O at origin, H1 at (0, 1.4307, 1.1078) Bohr, H2 at (0, -1.4307, 1.1078) Bohr.
/// Basis: STO-3G (Hehre, Stewart & Pople, JCP 51, 2657, 1969).
/// Shells: 0=O-1s, 1=O-2s, 2=O-2p, 3=H1-1s, 4=H2-1s.
pub fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];

    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // env[0..PTR_ENV_START] reserved for libcint global params (zeros = defaults).
    let mut env = vec![0.0_f64; PTR_ENV_START];

    let o_coord_ptr = env.len() as i32; // 20
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32; // 23
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32; // 26
    env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32; // 29
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32; // 30
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; // 33
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32; // 36
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; // 39
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32; // 42
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; // 45
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32; // 48
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; // 51
    env.extend_from_slice(&h_1s_coeff);

    // atm: O, H1, H2
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];

    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    // bas: O-1s, O-2s, O-2p, H1-1s, H2-1s
    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

/// Build H2O STO-3G fixture with PTR_F12_ZETA set for F12 oracle parity tests.
///
/// Sets `env[PTR_F12_ZETA]` (env[9]) to the given `zeta` value. This is required
/// for all F12/STG/YP integrals. A zeta of 0.0 must be explicitly rejected by the
/// cintx engine via `InvalidEnvParam`.
///
/// Typical value: `zeta = 1.2` (common F12 correlation factor exponent in production).
pub fn build_h2o_sto3g_f12(zeta: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    // PTR_F12_ZETA = 9 — within the PTR_ENV_START global params block.
    env[PTR_F12_ZETA] = zeta;
    (atm, bas, env)
}

pub const REQUIRED_MATRIX_ARTIFACT: &str =
    "/tmp/cintx_artifacts/cintx_phase_04_manifest_representation_matrix.json";
pub const MATRIX_ARTIFACT_FALLBACK_NAME: &str =
    "cintx_phase_04_manifest_representation_matrix.json";
pub const REQUIRED_REPORT_ARTIFACT: &str = "/tmp/cintx_artifacts/cintx_phase_04_compat_parity_report.json";
pub const REPORT_ARTIFACT_FALLBACK_NAME: &str = "cintx_phase_04_compat_parity_report.json";
pub const PHASE4_APPROVED_PROFILES: &[&str] =
    &["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"];
pub const PHASE4_ORACLE_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e", "4c1e"];
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileRepresentationMatrix {
    pub profile: String,
    pub fixtures: Vec<OracleFixture>,
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

#[derive(Clone, Debug)]
struct LockSymbolMetadata {
    profiles: BTreeSet<String>,
    stability: String,
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

/// Derive oracle-eligible families from the manifest lock.
/// Any entry with stability "stable" or "optional" is oracle-eligible.
/// Replaces the hardcoded PHASE4_ORACLE_FAMILIES constant.
pub fn manifest_oracle_families() -> BTreeSet<String> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON)
        .expect("compiled manifest lock JSON parse");
    root["entries"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|e| {
            let stab = e.get("stability").and_then(Value::as_str).unwrap_or("");
            if matches!(stab, "stable" | "optional") {
                e.get("id")
                    .and_then(|id| id.get("family"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Check if a family is oracle-eligible based on manifest or unstable-source prefix.
pub fn is_oracle_eligible_family(family: &str) -> bool {
    manifest_oracle_families().contains(family) || family.starts_with("unstable::source::")
}

fn is_phase4_oracle_family(family: &str) -> bool {
    is_oracle_eligible_family(family)
}

fn stability_is_included(stability: &str, include_unstable_source: bool) -> bool {
    match stability {
        "stable" | "optional" => true,
        "unstable_source" => include_unstable_source,
        _ => false,
    }
}

fn ensure_profile_approved(profile: &str) -> Result<()> {
    if PHASE4_APPROVED_PROFILES.contains(&profile) {
        return Ok(());
    }
    bail!(
        "unsupported profile `{profile}`; expected one of {:?}",
        PHASE4_APPROVED_PROFILES
    )
}

fn manifest_lock_symbol_metadata() -> Result<BTreeMap<String, LockSymbolMetadata>> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON)
        .context("parse compiled manifest lock")?;
    let entries = root
        .get("entries")
        .and_then(Value::as_array)
        .context("compiled manifest lock missing `entries` array")?;

    let mut symbols = BTreeMap::new();
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(Value::as_object)
            .context("compiled manifest entry missing `id`")?;
        let family = id.get("family").and_then(Value::as_str).unwrap_or_default();
        if !is_phase4_oracle_family(family) {
            continue;
        }
        let Some(symbol) = id.get("symbol").and_then(Value::as_str) else {
            continue;
        };
        let Some(_representation) = id.get("representation").and_then(Value::as_str) else {
            continue;
        };

        let profiles = entry
            .get("profiles")
            .and_then(Value::as_array)
            .map(|profiles| {
                profiles
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if profiles.is_empty() {
            continue;
        }
        let stability = entry
            .get("stability")
            .and_then(Value::as_str)
            .unwrap_or("stable")
            .to_owned();
        symbols.insert(
            symbol.to_owned(),
            LockSymbolMetadata {
                profiles,
                stability,
            },
        );
    }

    Ok(symbols)
}

fn phase4_operator_entries(
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<&'static ManifestEntry>> {
    ensure_profile_approved(profile)?;
    let metadata = manifest_lock_symbol_metadata()?;

    let mut entries = Vec::new();
    for entry in Resolver::manifest() {
        if !matches!(
            entry.helper_kind,
            HelperKind::Operator | HelperKind::SourceOnly
        ) {
            continue;
        }
        if !is_phase4_oracle_family(entry.family_name) {
            continue;
        }
        let Some(lock_entry) = metadata.get(entry.symbol_name) else {
            bail!(
                "manifest lock metadata missing for oracle symbol `{}`",
                entry.symbol_name
            );
        };
        if !lock_entry.profiles.contains(profile) {
            continue;
        }
        if !stability_is_included(&lock_entry.stability, include_unstable_source) {
            continue;
        }
        entries.push(entry);
    }
    entries.sort_by_key(|entry| entry.symbol_name);
    Ok(entries)
}

fn phase2_operator_entries() -> Result<Vec<&'static ManifestEntry>> {
    let entries = phase4_operator_entries(BASE_PROFILE, false)?;
    Ok(entries
        .into_iter()
        .filter(|entry| PHASE2_FAMILIES.contains(&entry.family_name))
        .collect())
}

pub fn phase2_manifest_symbols() -> BTreeSet<String> {
    phase2_operator_entries()
        .expect("phase2 manifest symbols")
        .into_iter()
        .map(|entry| entry.symbol_name.to_owned())
        .collect()
}

pub fn manifest_lock_symbols_for_profile(
    profile: &str,
    include_unstable_source: bool,
) -> Result<BTreeSet<String>> {
    ensure_profile_approved(profile)?;
    let metadata = manifest_lock_symbol_metadata()?;
    Ok(metadata
        .into_iter()
        .filter(|(_, value)| value.profiles.contains(profile))
        .filter(|(_, value)| stability_is_included(&value.stability, include_unstable_source))
        .map(|(symbol, _)| symbol)
        .collect())
}

pub fn manifest_lock_symbols() -> Result<BTreeSet<String>> {
    Ok(manifest_lock_symbols_for_profile(BASE_PROFILE, false)?
        .into_iter()
        .filter(|symbol| {
            Resolver::manifest().iter().any(|entry| {
                entry.symbol_name == symbol
                    && matches!(entry.helper_kind, HelperKind::Operator)
                    && PHASE2_FAMILIES.contains(&entry.family_name)
            })
        })
        .collect())
}

pub fn build_profile_representation_matrix(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<OracleFixture>> {
    let mut fixtures = Vec::new();
    for entry in phase4_operator_entries(profile, include_unstable_source)? {
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

pub fn build_required_profile_matrices(
    inputs: &OracleRawInputs,
) -> Result<Vec<ProfileRepresentationMatrix>> {
    PHASE4_APPROVED_PROFILES
        .iter()
        .copied()
        .map(|profile| {
            let fixtures = build_profile_representation_matrix(inputs, profile, false)?;
            Ok(ProfileRepresentationMatrix {
                profile: profile.to_owned(),
                fixtures,
            })
        })
        .collect()
}

pub fn build_phase2_representation_matrix(inputs: &OracleRawInputs) -> Result<Vec<OracleFixture>> {
    Ok(
        build_profile_representation_matrix(inputs, BASE_PROFILE, false)?
            .into_iter()
            .filter(|fixture| PHASE2_FAMILIES.contains(&fixture.family.as_str()))
            .collect(),
    )
}

pub fn write_profile_representation_matrix_artifact(
    profile: &str,
    include_unstable_source: bool,
    matrix: &[OracleFixture],
) -> Result<ArtifactWriteResult> {
    let artifact = build_matrix_artifact_json(profile, include_unstable_source, matrix)?;
    write_pretty_json_artifact(
        REQUIRED_MATRIX_ARTIFACT,
        MATRIX_ARTIFACT_FALLBACK_NAME,
        &artifact,
    )
}

fn build_matrix_artifact_json(
    profile: &str,
    include_unstable_source: bool,
    matrix: &[OracleFixture],
) -> Result<Value> {
    ensure_profile_approved(profile)?;

    let fixture_symbols: BTreeSet<&str> = matrix
        .iter()
        .map(|fixture| fixture.symbol.as_str())
        .collect();
    let expected_symbols = manifest_lock_symbols_for_profile(profile, include_unstable_source)?;
    let missing_symbols: Vec<String> = expected_symbols
        .iter()
        .filter(|symbol| !fixture_symbols.contains(symbol.as_str()))
        .cloned()
        .collect();
    if !missing_symbols.is_empty() {
        bail!(
            "fixture matrix for profile `{profile}` is missing {} symbols from compiled manifest lock",
            missing_symbols.len()
        );
    }

    let matrix_families: BTreeSet<&str> = matrix
        .iter()
        .map(|fixture| fixture.family.as_str())
        .collect();
    if matrix
        .iter()
        .any(|fixture| fixture.family.starts_with("unstable::source::"))
        && !include_unstable_source
    {
        bail!("fixture matrix unexpectedly contains unstable_source rows while disabled");
    }

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

    Ok(json!({
        "profile": profile,
        "include_unstable_source": include_unstable_source,
        "representation_matrix": fixtures_json,
        "fixture_count": matrix.len(),
        "required_path": REQUIRED_MATRIX_ARTIFACT,
        "compiled_manifest": "crates/cintx-ops/generated/compiled_manifest.lock.json",
        "approved_profiles": PHASE4_APPROVED_PROFILES,
        "oracle_families": manifest_oracle_families().into_iter().collect::<Vec<_>>(),
        "matrix_families": matrix_families,
    }))
}

pub fn write_representation_matrix_artifact(
    matrix: &[OracleFixture],
) -> Result<ArtifactWriteResult> {
    write_profile_representation_matrix_artifact(BASE_PROFILE, false, matrix)
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
    fn required_profile_matrices_match_manifest_profiles() {
        let inputs = OracleRawInputs::sample();
        let matrices = build_required_profile_matrices(&inputs).expect("required matrices");
        let actual_profiles: Vec<String> = matrices
            .iter()
            .map(|matrix| matrix.profile.clone())
            .collect();
        let expected_profiles: Vec<String> = PHASE4_APPROVED_PROFILES
            .iter()
            .map(|profile| (*profile).to_owned())
            .collect();
        assert_eq!(actual_profiles, expected_profiles);

        for matrix in matrices {
            let symbols: BTreeSet<String> = matrix
                .fixtures
                .iter()
                .map(|fixture| fixture.symbol.clone())
                .collect();
            let expected = manifest_lock_symbols_for_profile(&matrix.profile, false)
                .expect("profile lock symbols");
            assert_eq!(symbols, expected, "profile {} mismatch", matrix.profile);
        }
    }

    #[test]
    fn unstable_source_fixtures_require_opt_in() {
        let inputs = OracleRawInputs::sample();
        let stable_only =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, false).expect("stable");
        assert!(
            stable_only
                .iter()
                .all(|fixture| !fixture.family.starts_with("unstable::source::")),
            "stable run should exclude unstable_source fixtures"
        );

        let with_unstable =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, true).expect("unstable");
        assert!(
            with_unstable
                .iter()
                .any(|fixture| fixture.family.starts_with("unstable::source::")),
            "explicit unstable_source mode should include source-only fixtures"
        );
    }

    #[test]
    fn representation_matrix_artifact_is_written() {
        // Build the matrix and serialize it through the same code path as
        // write_representation_matrix_artifact, but write to an isolated temp
        // file to avoid races with parallel tests that share the fallback dir.
        let inputs = OracleRawInputs::sample();
        let matrix =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, false).expect("matrix");

        let tmp_dir = std::env::temp_dir().join(format!(
            "cintx_matrix_artifact_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = fs::create_dir_all(&tmp_dir);
        let artifact_path = tmp_dir.join(MATRIX_ARTIFACT_FALLBACK_NAME);

        let artifact =
            build_matrix_artifact_json(BASE_PROFILE, false, &matrix).expect("artifact json");
        let payload = serde_json::to_vec_pretty(&artifact).expect("serialize");
        fs::write(&artifact_path, &payload).expect("write artifact");

        assert!(artifact_path.is_file());
        let content = fs::read_to_string(&artifact_path).expect("artifact content");
        assert!(content.contains("representation_matrix"));
        assert!(content.contains(REQUIRED_MATRIX_ARTIFACT));
        assert!(content.contains("\"profile\": \"base\""));

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
