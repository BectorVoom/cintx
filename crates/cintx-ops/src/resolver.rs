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

    pub fn entries_by_kind(kind: HelperKind) -> Vec<&'static ManifestEntry> {
        MANIFEST_ENTRIES
            .iter()
            .filter(|entry| entry.helper_kind == kind)
            .collect()
    }

    pub fn filter_by_helper_kind(kind: HelperKind) -> Vec<&'static OperatorDescriptor> {
        OPERATOR_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.entry.helper_kind == kind)
            .collect()
    }

    pub fn helpers_by_kind(kind: HelperKind) -> Vec<&'static OperatorDescriptor> {
        Self::filter_by_helper_kind(kind)
            .into_iter()
            .filter(|descriptor| {
                descriptor.entry.category == "helper" || descriptor.entry.category == "legacy"
            })
            .collect()
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

    pub fn descriptor_by_symbol_and_kind(
        symbol: &str,
        helper_kind: HelperKind,
    ) -> Result<&'static OperatorDescriptor, ResolverError> {
        OPERATOR_DESCRIPTORS
            .iter()
            .find(|desc| desc.entry.symbol_name == symbol && desc.entry.helper_kind == helper_kind)
            .ok_or_else(|| ResolverError::MissingSymbol(symbol.to_owned()))
    }

    pub fn legacy_wrapper_from_misc(
        symbol: &str,
    ) -> Result<&'static OperatorDescriptor, ResolverError> {
        Self::descriptor_by_symbol_and_kind(symbol, HelperKind::Legacy)
    }

    pub fn resolve(
        family: &str,
        operator: &str,
        representation: Representation,
    ) -> Result<&'static OperatorDescriptor, ResolverError> {
        let matches: Vec<&OperatorDescriptor> = OPERATOR_DESCRIPTORS
            .iter()
            .filter(|desc| desc.entry.family_name == family && desc.entry.operator_name == operator)
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
    use std::collections::BTreeSet;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MiscWrapperMacro {
        AllCint,
        AllCint1e,
    }

    fn base_symbol_from_operator(entry: &ManifestEntry) -> Option<&'static str> {
        if !matches!(entry.helper_kind, HelperKind::Operator)
            || !entry.compiled_in_profiles.contains(&"base")
        {
            return None;
        }
        entry
            .symbol_name
            .strip_suffix("_cart")
            .or_else(|| entry.symbol_name.strip_suffix("_sph"))
            .or_else(|| entry.symbol_name.strip_suffix("_spinor"))
    }

    fn misc_wrapper_macro(base_symbol: &str) -> Option<MiscWrapperMacro> {
        match base_symbol {
            "int1e_ovlp"
            | "int1e_nuc"
            | "int2e"
            | "int2c2e"
            | "int3c1e_p2"
            | "int3c2e_ip1" => Some(MiscWrapperMacro::AllCint),
            "int1e_kin" => Some(MiscWrapperMacro::AllCint1e),
            _ => None,
        }
    }

    fn expected_legacy_wrapper_symbols(
        base_symbol: &str,
        macro_kind: MiscWrapperMacro,
    ) -> BTreeSet<String> {
        let mut expected = BTreeSet::from([
            format!("c{base_symbol}_cart"),
            format!("c{base_symbol}_sph"),
            format!("c{base_symbol}"),
        ]);
        if matches!(macro_kind, MiscWrapperMacro::AllCint) {
            expected.insert(format!("c{base_symbol}_cart_optimizer"));
            expected.insert(format!("c{base_symbol}_sph_optimizer"));
            expected.insert(format!("c{base_symbol}_optimizer"));
        }
        expected
    }

    use cintx_core::Representation;

    #[test]
    fn invalid_operator_id_is_reported() {
        let raw = u32::MAX;
        let err = Resolver::descriptor(OperatorId::new(raw)).unwrap_err();
        assert!(matches!(err, ResolverError::MissingOperatorId(id) if id == raw));
    }

    #[test]
    fn resolve_uses_metadata_over_symbol() {
        let descriptor = Resolver::resolve("1e", "overlap", Representation::Cart)
            .expect("metadata should exist");
        assert_eq!(descriptor.operator_name(), "overlap");
        assert_eq!(descriptor.family(), "1e");
        assert!(
            descriptor
                .entry
                .supports_representation(Representation::Cart)
        );
    }

    #[test]
    fn missing_operator_is_reported() {
        let err = Resolver::resolve("1e", "missing", Representation::Cart).unwrap_err();
        assert!(matches!(err, ResolverError::MissingFamilyOperator { .. }));
    }

    #[test]
    fn helpers_by_kind_filters_entries() {
        let transforms = Resolver::helpers_by_kind(HelperKind::Transform);
        assert!(!transforms.is_empty());
        assert!(
            transforms
                .iter()
                .all(|descriptor| matches!(descriptor.entry.helper_kind, HelperKind::Transform))
        );
    }

    #[test]
    fn legacy_wrapper_lookup_requires_legacy_kind() {
        let descriptor =
            Resolver::legacy_wrapper_from_misc("cint2e_cart_optimizer").expect("legacy wrapper");
        assert!(matches!(descriptor.entry.helper_kind, HelperKind::Legacy));

        let err = Resolver::legacy_wrapper_from_misc("int2e_cart").unwrap_err();
        assert!(matches!(err, ResolverError::MissingSymbol(symbol) if symbol == "int2e_cart"));
    }

    #[test]
    fn legacy_wrapper_manifest_matches_misc() {
        let mut base_symbols = BTreeSet::new();
        for entry in Resolver::entries_by_kind(HelperKind::Operator) {
            if let Some(base_symbol) = base_symbol_from_operator(entry) {
                base_symbols.insert(base_symbol.to_owned());
            }
        }

        let mut expected = BTreeSet::new();
        for base_symbol in base_symbols {
            let macro_kind = misc_wrapper_macro(&base_symbol).unwrap_or_else(|| {
                panic!("missing misc.h wrapper macro classification for {base_symbol}")
            });
            expected.extend(expected_legacy_wrapper_symbols(&base_symbol, macro_kind));
        }

        let actual: BTreeSet<String> = Resolver::entries_by_kind(HelperKind::Legacy)
            .into_iter()
            .filter(|entry| entry.compiled_in_profiles.contains(&"base"))
            .map(|entry| entry.symbol_name.to_owned())
            .collect();

        assert_eq!(
            actual, expected,
            "legacy wrapper entries drifted from misc.h wrapper rules"
        );
    }
}
