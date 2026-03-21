use crate::generated::{MANIFEST_ENTRIES, OPERATOR_DESCRIPTORS};
use cintx_core::{OperatorId, Representation};
use std::borrow::Cow;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureFlag {
    None,
    WithF12,
    With4c1e,
    WithF12With4c1e,
    UnstableSource,
    Other(Cow<'static, str>),
}

impl FeatureFlag {
    pub fn from_name(value: &str) -> Self {
        match value {
            "with-f12" => FeatureFlag::WithF12,
            "with-4c1e" => FeatureFlag::With4c1e,
            "with-f12+with-4c1e" => FeatureFlag::WithF12With4c1e,
            "unstable_source" => FeatureFlag::UnstableSource,
            "" | "none" => FeatureFlag::None,
            other => FeatureFlag::Other(Cow::Owned(other.to_owned())),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stability {
    Stable,
    Optional,
    UnstableSource,
    Other(Cow<'static, str>),
}

impl Stability {
    pub fn from_name(value: &str) -> Self {
        match value {
            "stable" => Stability::Stable,
            "optional" => Stability::Optional,
            "unstable_source" => Stability::UnstableSource,
            other => Stability::Other(Cow::Owned(other.to_owned())),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HelperKind {
    Operator,
    Helper,
    Transform,
    Optimizer,
    Legacy,
    SourceOnly,
    Other(Cow<'static, str>),
}

impl HelperKind {
    pub fn from_name(value: &str) -> Self {
        match value {
            "helper" => HelperKind::Helper,
            "transform" => HelperKind::Transform,
            "optimizer" => HelperKind::Optimizer,
            "legacy" => HelperKind::Legacy,
            "source" | "source_only" | "source-only" => HelperKind::SourceOnly,
            "operator" => HelperKind::Operator,
            other => HelperKind::Other(Cow::Owned(other.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepresentationSupport {
    pub cart: bool,
    pub spheric: bool,
    pub spinor: bool,
}

impl RepresentationSupport {
    pub const fn new(cart: bool, spheric: bool, spinor: bool) -> Self {
        RepresentationSupport {
            cart,
            spheric,
            spinor,
        }
    }

    pub const fn supports(&self, rep: Representation) -> bool {
        match rep {
            Representation::Cart => self.cart,
            Representation::Spheric => self.spheric,
            Representation::Spinor => self.spinor,
        }
    }
}

#[derive(Debug)]
pub struct ManifestEntry {
    pub family_name: &'static str,
    pub operator_name: &'static str,
    pub symbol_name: &'static str,
    pub category: &'static str,
    pub arity: u8,
    pub forms: &'static [&'static str],
    pub component_rank: &'static str,
    pub feature_flag: FeatureFlag,
    pub stability: Stability,
    pub declared_in: &'static str,
    pub compiled_in_profiles: &'static [&'static str],
    pub oracle_covered: bool,
    pub helper_kind: HelperKind,
    pub canonical_family: &'static str,
    pub representation: RepresentationSupport,
}

impl ManifestEntry {
    pub fn supports_representation(&self, rep: Representation) -> bool {
        self.representation.supports(rep)
    }
}

#[derive(Debug)]
pub struct OperatorDescriptor {
    pub id: OperatorId,
    pub entry: &'static ManifestEntry,
}

impl OperatorDescriptor {
    pub fn family(&self) -> &'static str {
        self.entry.family_name
    }

    pub fn operator_symbol(&self) -> &'static str {
        self.entry.symbol_name
    }

    pub fn operator_name(&self) -> &'static str {
        self.entry.operator_name
    }

    pub fn feature_flag(&self) -> FeatureFlag {
        self.entry.feature_flag.clone()
    }

    pub fn stability(&self) -> Stability {
        self.entry.stability.clone()
    }
}

#[derive(Debug, Error)]
pub enum ResolverError {
    #[error("operator id {0} is missing from manifest")]
    MissingOperatorId(u32),
    #[error("symbol {0} is not part of the canonical manifest lock")]
    MissingSymbol(String),
    #[error("family {family} operator {operator} not found in manifest")]
    MissingFamilyOperator { family: String, operator: String },
    #[error("family {family} operator {operator} does not support {representation:?}")]
    UnsupportedRepresentation {
        family: String,
        operator: String,
        representation: Representation,
    },
}

pub struct Resolver;

impl Resolver {
    pub fn manifest() -> &'static [ManifestEntry] {
        MANIFEST_ENTRIES
    }

    pub fn descriptors() -> &'static [OperatorDescriptor] {
        OPERATOR_DESCRIPTORS
    }

    pub fn descriptor(id: OperatorId) -> Result<&'static OperatorDescriptor, ResolverError> {
        OPERATOR_DESCRIPTORS
            .get(id.raw() as usize)
            .ok_or(ResolverError::MissingOperatorId(id.raw()))
    }

    pub fn descriptor_by_symbol(
        symbol: &str,
    ) -> Result<&'static OperatorDescriptor, ResolverError> {
        OPERATOR_DESCRIPTORS
            .iter()
            .find(|desc| desc.entry.symbol_name == symbol)
            .ok_or_else(|| ResolverError::MissingSymbol(symbol.to_owned()))
    }

    pub fn resolve(
        family: &str,
        operator: &str,
        representation: Representation,
    ) -> Result<&'static OperatorDescriptor, ResolverError> {
        let matches: Vec<&OperatorDescriptor> = OPERATOR_DESCRIPTORS
            .iter()
            .filter(|desc| {
                desc.entry.family_name == family && desc.entry.operator_name == operator
            })
            .collect();

        if matches.is_empty() {
            return Err(ResolverError::MissingFamilyOperator {
                family: family.to_string(),
                operator: operator.to_string(),
            });
        }

        matches
            .into_iter()
            .find(|desc| desc.entry.supports_representation(representation))
            .ok_or_else(|| ResolverError::UnsupportedRepresentation {
                family: family.to_string(),
                operator: operator.to_string(),
                representation,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::Representation;

    #[test]
    fn invalid_operator_id_is_reported() {
        let raw = u32::MAX;
        let err = Resolver::descriptor(OperatorId::new(raw)).unwrap_err();
        assert!(matches!(err, ResolverError::MissingOperatorId(id) if id == raw));
    }

    #[test]
    fn resolve_uses_metadata_over_symbol() {
        let descriptor =
            Resolver::resolve("1e", "overlap", Representation::Cart).expect("metadata should exist");
        assert_eq!(descriptor.operator_name(), "overlap");
        assert_eq!(descriptor.family(), "1e");
        assert!(descriptor.entry.supports_representation(Representation::Cart));
    }

    #[test]
    fn missing_operator_is_reported() {
        let err = Resolver::resolve("1e", "missing", Representation::Cart).unwrap_err();
        assert!(matches!(err, ResolverError::MissingFamilyOperator { .. }));
    }
}
