use super::canonicalize::{canonicalize_profile_label, canonicalize_symbol_name};
use crate::contracts::{IntegralFamily, OperatorKind, Representation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ManifestProfile {
    #[serde(rename = "base")]
    Base,
    #[serde(rename = "with-f12")]
    WithF12,
    #[serde(rename = "with-4c1e")]
    With4c1e,
    #[serde(rename = "with-f12+with-4c1e")]
    WithF12With4c1e,
}

impl ManifestProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::WithF12 => "with-f12",
            Self::With4c1e => "with-4c1e",
            Self::WithF12With4c1e => "with-f12+with-4c1e",
        }
    }

    pub fn approved_scope() -> Vec<Self> {
        vec![
            Self::Base,
            Self::WithF12,
            Self::With4c1e,
            Self::WithF12With4c1e,
        ]
    }

    pub fn parse(raw: &str) -> Result<Self, ManifestGovernanceError> {
        let normalized = canonicalize_profile_label(raw);
        match normalized.as_str() {
            "base" => Ok(Self::Base),
            "with-f12" | "f12" => Ok(Self::WithF12),
            "with-4c1e" | "4c1e" => Ok(Self::With4c1e),
            "with-f12+with-4c1e" | "with-4c1e+with-f12" | "f12+4c1e" | "4c1e+f12" => {
                Ok(Self::WithF12With4c1e)
            }
            _ => Err(ManifestGovernanceError::UnknownProfile {
                provided: raw.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StabilityClass {
    #[serde(rename = "stable")]
    Stable,
    #[serde(rename = "experimental")]
    Experimental,
    #[serde(rename = "deprecated")]
    Deprecated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FamilyTag {
    #[serde(rename = "1e")]
    OneElectron,
    #[serde(rename = "2e")]
    TwoElectron,
    #[serde(rename = "2c2e")]
    TwoCenterTwoElectron,
    #[serde(rename = "3c1e")]
    ThreeCenterOneElectron,
    #[serde(rename = "3c2e")]
    ThreeCenterTwoElectron,
    #[serde(rename = "4c1e")]
    FourCenterOneElectron,
}

impl FamilyTag {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneElectron => "1e",
            Self::TwoElectron => "2e",
            Self::TwoCenterTwoElectron => "2c2e",
            Self::ThreeCenterOneElectron => "3c1e",
            Self::ThreeCenterTwoElectron => "3c2e",
            Self::FourCenterOneElectron => "4c1e",
        }
    }
}

impl From<IntegralFamily> for FamilyTag {
    fn from(value: IntegralFamily) -> Self {
        match value {
            IntegralFamily::OneElectron => Self::OneElectron,
            IntegralFamily::TwoElectron => Self::TwoElectron,
            IntegralFamily::TwoCenterTwoElectron => Self::TwoCenterTwoElectron,
            IntegralFamily::ThreeCenterOneElectron => Self::ThreeCenterOneElectron,
            IntegralFamily::ThreeCenterTwoElectron => Self::ThreeCenterTwoElectron,
            IntegralFamily::FourCenterOneElectron => Self::FourCenterOneElectron,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatorTag {
    #[serde(rename = "overlap")]
    Overlap,
    #[serde(rename = "kinetic")]
    Kinetic,
    #[serde(rename = "nuclear-attraction")]
    NuclearAttraction,
    #[serde(rename = "electron-repulsion")]
    ElectronRepulsion,
}

impl OperatorTag {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overlap => "overlap",
            Self::Kinetic => "kinetic",
            Self::NuclearAttraction => "nuclear-attraction",
            Self::ElectronRepulsion => "electron-repulsion",
        }
    }
}

impl From<OperatorKind> for OperatorTag {
    fn from(value: OperatorKind) -> Self {
        match value {
            OperatorKind::Overlap => Self::Overlap,
            OperatorKind::Kinetic => Self::Kinetic,
            OperatorKind::NuclearAttraction => Self::NuclearAttraction,
            OperatorKind::ElectronRepulsion => Self::ElectronRepulsion,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RepresentationTag {
    #[serde(rename = "cart")]
    Cartesian,
    #[serde(rename = "sph")]
    Spherical,
    #[serde(rename = "spinor")]
    Spinor,
}

impl RepresentationTag {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cartesian => "cart",
            Self::Spherical => "sph",
            Self::Spinor => "spinor",
        }
    }
}

impl From<Representation> for RepresentationTag {
    fn from(value: Representation) -> Self {
        match value {
            Representation::Cartesian => Self::Cartesian,
            Representation::Spherical => Self::Spherical,
            Representation::Spinor => Self::Spinor,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalSymbolIdentity {
    pub family: FamilyTag,
    pub operator: OperatorTag,
    pub representation: RepresentationTag,
    pub symbol: String,
}

impl CanonicalSymbolIdentity {
    pub fn new(
        family: IntegralFamily,
        operator: OperatorKind,
        representation: Representation,
        symbol: impl AsRef<str>,
    ) -> Result<Self, ManifestGovernanceError> {
        let canonical_symbol = canonicalize_symbol_name(symbol.as_ref());
        if canonical_symbol.is_empty() {
            return Err(ManifestGovernanceError::EmptySymbolIdentity);
        }
        Ok(Self {
            family: family.into(),
            operator: operator.into(),
            representation: representation.into(),
            symbol: canonical_symbol,
        })
    }

    fn key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.family.as_str(),
            self.operator.as_str(),
            self.representation.as_str(),
            self.symbol
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestLockEntry {
    pub id: CanonicalSymbolIdentity,
    pub profiles: Vec<ManifestProfile>,
    pub stability: StabilityClass,
}

impl ManifestLockEntry {
    pub fn new(
        id: CanonicalSymbolIdentity,
        profiles: impl IntoIterator<Item = ManifestProfile>,
        stability: StabilityClass,
    ) -> Result<Self, ManifestGovernanceError> {
        let normalized_profiles = normalize_profiles(profiles);
        if normalized_profiles.is_empty() {
            return Err(ManifestGovernanceError::EmptyProfileMembership { symbol: id.key() });
        }
        Ok(Self {
            id,
            profiles: normalized_profiles,
            stability,
        })
    }

    pub fn from_profile_labels<I, S>(
        id: CanonicalSymbolIdentity,
        profile_labels: I,
        stability: StabilityClass,
    ) -> Result<Self, ManifestGovernanceError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let parsed = profile_labels
            .into_iter()
            .map(|label| ManifestProfile::parse(label.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(id, parsed, stability)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileScopeMetadata {
    pub approved: Vec<ManifestProfile>,
    pub observed_union: Vec<ManifestProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledManifestLock {
    pub schema_version: u32,
    pub profile_scope: ProfileScopeMetadata,
    pub entries: Vec<ManifestLockEntry>,
}

impl CompiledManifestLock {
    pub fn new(entries: Vec<ManifestLockEntry>) -> Result<Self, ManifestGovernanceError> {
        let mut lock = Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            profile_scope: ProfileScopeMetadata {
                approved: ManifestProfile::approved_scope(),
                observed_union: Vec::new(),
            },
            entries,
        };
        lock.normalize_entries();
        lock.recompute_profile_union();
        lock.validate_schema_invariants()?;
        Ok(lock)
    }

    pub fn validate_schema_invariants(&self) -> Result<(), ManifestGovernanceError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(ManifestGovernanceError::SchemaVersionMismatch {
                expected: MANIFEST_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        let approved = approved_profile_set();
        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            if entry.id.symbol.is_empty() {
                return Err(ManifestGovernanceError::EmptySymbolIdentity);
            }
            if entry.profiles.is_empty() {
                return Err(ManifestGovernanceError::EmptyProfileMembership {
                    symbol: entry.id.key(),
                });
            }
            let key = entry.id.key();
            if !seen.insert(key.clone()) {
                return Err(ManifestGovernanceError::DuplicateCanonicalEntry { symbol: key });
            }
            for profile in &entry.profiles {
                if !approved.contains(profile) {
                    return Err(ManifestGovernanceError::ProfileOutsideApprovedScope {
                        symbol: entry.id.key(),
                        profile: profile.as_str().to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn profile_union(&self) -> Vec<ManifestProfile> {
        self.profile_scope.observed_union.clone()
    }

    pub fn validate_profile_union(&self) -> Result<(), ManifestGovernanceError> {
        let expected = ManifestProfile::approved_scope();
        let actual = self.profile_union();
        if actual != expected {
            return Err(ManifestGovernanceError::ProfileUnionDrift {
                expected: expected
                    .into_iter()
                    .map(ManifestProfile::as_str)
                    .map(str::to_string)
                    .collect(),
                actual: actual
                    .into_iter()
                    .map(ManifestProfile::as_str)
                    .map(str::to_string)
                    .collect(),
            });
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<String, ManifestGovernanceError> {
        Ok(serde_json::to_string_pretty(&self.canonicalized())?)
    }

    pub fn enforce_drift_policy(
        &self,
        baseline: &Self,
        approval: Option<&LockUpdateApproval>,
    ) -> Result<(), ManifestGovernanceError> {
        let baseline_json = baseline.canonical_json()?;
        let current_json = self.canonical_json()?;
        if baseline_json == current_json {
            return Ok(());
        }
        match approval {
            Some(approval) => {
                approval.validate()?;
                Ok(())
            }
            None => Err(ManifestGovernanceError::UnapprovedLockDrift {
                expected: baseline_json,
                actual: current_json,
            }),
        }
    }

    fn canonicalized(&self) -> Self {
        let mut canonical = self.clone();
        canonical.normalize_entries();
        canonical.recompute_profile_union();
        canonical
    }

    fn normalize_entries(&mut self) {
        for entry in &mut self.entries {
            entry.id.symbol = canonicalize_symbol_name(&entry.id.symbol);
            entry.profiles = normalize_profiles(entry.profiles.iter().copied());
        }
        self.entries.sort_by(|left, right| {
            (
                left.id.family,
                left.id.operator,
                left.id.representation,
                left.id.symbol.as_str(),
            )
                .cmp(&(
                    right.id.family,
                    right.id.operator,
                    right.id.representation,
                    right.id.symbol.as_str(),
                ))
        });
    }

    fn recompute_profile_union(&mut self) {
        self.profile_scope.observed_union = normalize_profiles(
            self.entries
                .iter()
                .flat_map(|entry| entry.profiles.iter().copied()),
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockUpdateReason {
    #[serde(rename = "schema-change")]
    SchemaChange,
    #[serde(rename = "upstream-symbol-change")]
    UpstreamSymbolChange,
    #[serde(rename = "profile-policy-change")]
    ProfilePolicyChange,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockUpdateApproval {
    pub reason: LockUpdateReason,
    pub rationale: String,
}

impl LockUpdateApproval {
    pub fn new(reason: LockUpdateReason, rationale: impl Into<String>) -> Self {
        Self {
            reason,
            rationale: rationale.into(),
        }
    }

    fn validate(&self) -> Result<(), ManifestGovernanceError> {
        if self.rationale.trim().is_empty() {
            return Err(ManifestGovernanceError::EmptyApprovalRationale);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ManifestGovernanceError {
    #[error("manifest lock schema version mismatch: expected {expected}, got {actual}")]
    SchemaVersionMismatch { expected: u32, actual: u32 },
    #[error("canonical symbol identity cannot be empty")]
    EmptySymbolIdentity,
    #[error("manifest entry `{symbol}` has no profile membership")]
    EmptyProfileMembership { symbol: String },
    #[error("manifest profile `{provided}` is not recognized")]
    UnknownProfile { provided: String },
    #[error("manifest entry `{symbol}` is duplicated in canonical lock")]
    DuplicateCanonicalEntry { symbol: String },
    #[error("manifest entry `{symbol}` includes unapproved profile `{profile}`")]
    ProfileOutsideApprovedScope { symbol: String, profile: String },
    #[error("profile-union drift detected: expected {expected:?}, actual {actual:?}")]
    ProfileUnionDrift {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    #[error("unapproved compiled-manifest lock drift detected")]
    UnapprovedLockDrift { expected: String, actual: String },
    #[error("compiled-manifest audit invariant failed: {detail}")]
    AuditInvariant { detail: String },
    #[error("compiled-manifest feature flag `{feature_flag}` is not recognized")]
    UnknownFeatureFlag { feature_flag: String },
    #[error("lock update approval rationale cannot be empty")]
    EmptyApprovalRationale,
    #[error("failed to encode/decode canonical manifest lock: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn approved_profile_set() -> BTreeSet<ManifestProfile> {
    ManifestProfile::approved_scope().into_iter().collect()
}

fn normalize_profiles(profiles: impl IntoIterator<Item = ManifestProfile>) -> Vec<ManifestProfile> {
    let set: BTreeSet<ManifestProfile> = profiles.into_iter().collect();
    set.into_iter().collect()
}
