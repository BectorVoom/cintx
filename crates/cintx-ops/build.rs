use serde::Deserialize;
use std::collections::HashSet;
use std::fmt::Write;
use std::fs;
use std::path::Path;

const EXPECTED_PROFILES: &[&str] = &["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"];

fn main() {
    if let Err(err) = try_generate_manifest() {
        panic!("failed to generate manifest: {err}");
    }
}

fn try_generate_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let canonical_path = Path::new("generated/compiled_manifest.lock.json");
    println!("cargo:rerun-if-changed={}", canonical_path.display());
    let payload = fs::read_to_string(canonical_path)?;
    let lock: LockFile = serde_json::from_str(&payload)?;
    validate_profile_scope(&lock.profile_scope)?;

    let entries: Vec<GeneratedEntry> = lock.entries.iter().map(GeneratedEntry::from).collect();

    let mut rs_buffer = String::new();
    writeln!(rs_buffer, "// Generated manifest; do not edit.")?;
    writeln!(
        rs_buffer,
        "pub const MANIFEST_SCHEMA_VERSION: u32 = {};",
        lock.schema_version
    )?;

    let approved_literals = if lock.profile_scope.approved.is_empty() {
        "&[]".to_string()
    } else {
        let joined = lock
            .profile_scope
            .approved
            .iter()
            .map(|value| literal(value))
            .collect::<Vec<_>>()
            .join(", ");
        format!("&[{}]", joined)
    };
    writeln!(
        rs_buffer,
        "pub const PROFILE_SCOPE_APPROVED: &[&str] = {};",
        approved_literals
    )?;

    writeln!(
        rs_buffer,
        "use crate::resolver::{{FeatureFlag, HelperKind, ManifestEntry, OperatorDescriptor, RepresentationSupport, Stability}};"
    )?;
    writeln!(rs_buffer, "use cintx_core::OperatorId;")?;
    writeln!(
        rs_buffer,
        "pub const MANIFEST_ENTRIES: &[ManifestEntry] = &["
    )?;

    for entry in &entries {
        let forms_literal =
            array_literal(&entry.forms.iter().map(|v| v.as_str()).collect::<Vec<_>>());
        let compiled_literal = array_literal(
            &entry
                .compiled_in_profiles
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>(),
        );
        let family_literal = literal(&entry.family);
        let operator_literal = literal(&entry.operator);
        let symbol_literal = literal(&entry.symbol);
        let category_literal = literal(&entry.category);
        let component_literal = literal(&entry.component_rank);
        let declared_literal = literal(&entry.declared_in);
        let canonical_literal = literal(&entry.canonical_family);
        let feature_flag = feature_flag_snippet(&entry.feature_flag);
        let stability = stability_snippet(&entry.stability);
        let helper = helper_kind_snippet(&entry.helper_kind);
        writeln!(
            rs_buffer,
            "    ManifestEntry {{ family_name: {family}, operator_name: {operator}, symbol_name: {symbol}, category: {category}, arity: {arity}, forms: {forms}, component_rank: {component}, feature_flag: {feature_flag}, stability: {stability}, declared_in: {declared}, compiled_in_profiles: {compiled}, oracle_covered: {oracle}, helper_kind: {helper}, canonical_family: {canonical}, representation: RepresentationSupport::new({cart}, {spheric}, {spinor}) }},",
            family = family_literal,
            operator = operator_literal,
            symbol = symbol_literal,
            category = category_literal,
            arity = entry.arity,
            forms = forms_literal,
            component = component_literal,
            feature_flag = feature_flag,
            stability = stability,
            declared = declared_literal,
            compiled = compiled_literal,
            oracle = entry.oracle_covered,
            helper = helper,
            canonical = canonical_literal,
            cart = entry.rep_cart,
            spheric = entry.rep_spheric,
            spinor = entry.rep_spinor,
        )?;
    }

    writeln!(rs_buffer, "];\n")?;
    writeln!(
        rs_buffer,
        "pub const OPERATOR_DESCRIPTORS: &[OperatorDescriptor] = &["
    )?;
    for idx in 0..entries.len() {
        writeln!(
            rs_buffer,
            "    OperatorDescriptor {{ id: OperatorId::new({idx}), entry: &MANIFEST_ENTRIES[{idx}] }},",
            idx = idx,
        )?;
    }
    writeln!(rs_buffer, "];\n")?;

    let mut csv_buffer = String::from(
        "family_name,operator_name,symbol_name,category,arity,forms,component_rank,feature_flag,stability,declared_in,compiled_in_profiles,oracle_covered,helper_kind,canonical_family\n",
    );
    for entry in &entries {
        writeln!(
            csv_buffer,
            "{family},{operator},{symbol},{category},{arity},{forms},{component},{feature_flag},{stability},{declared},{compiled},{oracle},{helper},{canonical}",
            family = csv_quote(&entry.family),
            operator = csv_quote(&entry.operator),
            symbol = csv_quote(&entry.symbol),
            category = csv_quote(&entry.category),
            arity = entry.arity,
            forms = csv_quote(&entry.forms.join("|")),
            component = csv_quote(&entry.component_rank),
            feature_flag = csv_quote(&entry.feature_flag),
            stability = csv_quote(&entry.stability),
            declared = csv_quote(&entry.declared_in),
            compiled = csv_quote(&entry.compiled_in_profiles.join("|")),
            oracle = entry.oracle_covered,
            helper = csv_quote(&entry.helper_kind),
            canonical = csv_quote(&entry.canonical_family),
        )?;
    }

    fs::write(Path::new("src/generated/api_manifest.rs"), rs_buffer)?;
    fs::write(Path::new("src/generated/api_manifest.csv"), csv_buffer)?;

    Ok(())
}

fn literal(input: &str) -> String {
    format!("{:?}", input)
}

fn array_literal(values: &[&str]) -> String {
    if values.is_empty() {
        "&[]".to_string()
    } else {
        let joined = values
            .iter()
            .map(|value| format!("{:?}", value))
            .collect::<Vec<_>>()
            .join(", ");
        format!("&[{}]", joined)
    }
}

fn feature_flag_snippet(flag: &str) -> String {
    match flag {
        "with-f12" => "FeatureFlag::WithF12".to_string(),
        "with-4c1e" => "FeatureFlag::With4c1e".to_string(),
        "with-f12+with-4c1e" => "FeatureFlag::WithF12With4c1e".to_string(),
        "unstable_source" => "FeatureFlag::UnstableSource".to_string(),
        "" | "none" => "FeatureFlag::None".to_string(),
        other => format!(
            "FeatureFlag::Other(std::borrow::Cow::Borrowed({}))",
            literal(other)
        ),
    }
}

fn stability_snippet(value: &str) -> String {
    match value {
        "stable" => "Stability::Stable".to_string(),
        "optional" => "Stability::Optional".to_string(),
        "unstable_source" => "Stability::UnstableSource".to_string(),
        other => format!(
            "Stability::Other(std::borrow::Cow::Borrowed({}))",
            literal(other)
        ),
    }
}

fn helper_kind_snippet(value: &str) -> String {
    match value {
        "helper" => "HelperKind::Helper".to_string(),
        "transform" => "HelperKind::Transform".to_string(),
        "optimizer" => "HelperKind::Optimizer".to_string(),
        "legacy" => "HelperKind::Legacy".to_string(),
        "source" | "source_only" | "source-only" => "HelperKind::SourceOnly".to_string(),
        "operator" => "HelperKind::Operator".to_string(),
        other => format!(
            "HelperKind::Other(std::borrow::Cow::Borrowed({}))",
            literal(other)
        ),
    }
}

fn csv_quote(value: &str) -> String {
    let escaped = value.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

fn entry_forms(entry: &LockEntry) -> Vec<String> {
    if let Some(forms) = entry.forms.as_ref() {
        return forms.clone();
    }
    entry
        .id
        .representation
        .as_ref()
        .map(|value| vec![value.clone()])
        .unwrap_or_default()
}

fn representation_flags(forms: &[String], representation: Option<&str>) -> (bool, bool, bool) {
    let mut cart = false;
    let mut spheric = false;
    let mut spinor = false;

    for form in forms {
        match form.as_str() {
            "cart" => cart = true,
            "sph" | "spheric" => spheric = true,
            "spinor" => spinor = true,
            _ => {}
        }
    }

    if !cart && matches!(representation, Some(rep) if rep == "cart") {
        cart = true;
    }
    if !spheric && matches!(representation, Some(rep) if rep == "sph" || rep == "spheric") {
        spheric = true;
    }
    if !spinor && matches!(representation, Some(rep) if rep == "spinor") {
        spinor = true;
    }

    (cart, spheric, spinor)
}

fn family_arity(family: &str) -> u8 {
    match family {
        "1e" => 2,
        "2c2e" => 2,
        "3c1e" => 3,
        "3c2e" => 3,
        "2e" | "4c1e" => 4,
        _ => 0,
    }
}

fn validate_profile_scope(scope: &ProfileScope) -> Result<(), String> {
    let seen: HashSet<_> = scope.observed_union.iter().map(|s| s.as_str()).collect();
    let expected: HashSet<_> = EXPECTED_PROFILES.iter().copied().collect();
    if seen != expected {
        Err(format!(
            "observed profile union does not match expected set: {seen:?} vs {expected:?}"
        ))
    } else {
        Ok(())
    }
}

#[derive(Deserialize)]
struct LockFile {
    schema_version: u32,
    profile_scope: ProfileScope,
    entries: Vec<LockEntry>,
}

#[derive(Deserialize)]
struct ProfileScope {
    approved: Vec<String>,
    observed_union: Vec<String>,
}

#[derive(Deserialize)]
struct LockEntry {
    id: EntryId,
    profiles: Vec<String>,
    stability: Option<String>,
    category: Option<String>,
    arity: Option<u8>,
    forms: Option<Vec<String>>,
    component_rank: Option<String>,
    feature_flag: Option<String>,
    declared_in: Option<String>,
    compiled_in_profiles: Option<Vec<String>>,
    oracle_covered: Option<bool>,
    helper_kind: Option<String>,
    canonical_family: Option<String>,
}

#[derive(Deserialize)]
struct EntryId {
    family: String,
    operator: Option<String>,
    representation: Option<String>,
    symbol: String,
}

struct GeneratedEntry {
    family: String,
    operator: String,
    symbol: String,
    category: String,
    arity: u8,
    forms: Vec<String>,
    component_rank: String,
    feature_flag: String,
    stability: String,
    declared_in: String,
    compiled_in_profiles: Vec<String>,
    oracle_covered: bool,
    helper_kind: String,
    canonical_family: String,
    rep_cart: bool,
    rep_spheric: bool,
    rep_spinor: bool,
}

impl From<&LockEntry> for GeneratedEntry {
    fn from(entry: &LockEntry) -> Self {
        let family = entry.id.family.clone();
        let operator = entry
            .id
            .operator
            .clone()
            .unwrap_or_else(|| entry.id.symbol.clone());
        let forms = entry_forms(entry);
        let (rep_cart, rep_spheric, rep_spinor) =
            representation_flags(&forms, entry.id.representation.as_deref());
        let compiled_profiles = entry
            .compiled_in_profiles
            .clone()
            .unwrap_or_else(|| entry.profiles.clone());
        GeneratedEntry {
            family: family.clone(),
            operator,
            symbol: entry.id.symbol.clone(),
            category: entry.category.clone().unwrap_or_else(|| family.clone()),
            arity: entry.arity.unwrap_or_else(|| family_arity(&family)),
            forms,
            component_rank: entry.component_rank.clone().unwrap_or_default(),
            feature_flag: entry.feature_flag.clone().unwrap_or_else(|| "none".into()),
            stability: entry.stability.clone().unwrap_or_else(|| "stable".into()),
            declared_in: entry
                .declared_in
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            compiled_in_profiles: compiled_profiles,
            oracle_covered: entry.oracle_covered.unwrap_or(true),
            helper_kind: entry
                .helper_kind
                .clone()
                .unwrap_or_else(|| "operator".into()),
            canonical_family: entry
                .canonical_family
                .clone()
                .unwrap_or_else(|| family.clone()),
            rep_cart,
            rep_spheric,
            rep_spinor,
        }
    }
}
