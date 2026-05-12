use std::fmt;

/// Supported symmetry representations for libcint operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Representation {
    Cart,
    Spheric,
    Spinor,
}

impl fmt::Display for Representation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Representation::Cart => write!(f, "Cart"),
            Representation::Spheric => write!(f, "Spheric"),
            Representation::Spinor => write!(f, "Spinor"),
        }
    }
}

/// AO symmetry packing convention (pyscf-compatible naming).
///
/// Phase 18 ships `S1` only; every other variant returns
/// `FacadeError::UnsupportedAoSymmetry` from `SessionRequest::query_workspace`.
/// `Display` emits the lowercase pyscf form (`s1`, `s2ij`, `s2kl`, `s4`, `s8`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AoSymmetry {
    #[default]
    S1,
    S2ij,
    S2kl,
    S4,
    S8,
}

impl fmt::Display for AoSymmetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AoSymmetry::S1 => write!(f, "s1"),
            AoSymmetry::S2ij => write!(f, "s2ij"),
            AoSymmetry::S2kl => write!(f, "s2kl"),
            AoSymmetry::S4 => write!(f, "s4"),
            AoSymmetry::S8 => write!(f, "s8"),
        }
    }
}

/// Lean wrapper around the generated operator index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct OperatorId(u32);

impl OperatorId {
    pub const fn new(raw: u32) -> Self {
        OperatorId(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    // Phase 19 D-09: typed constants for the four `int1e_ecp_*` operators.
    // Values are derived from crates/cintx-ops/src/generated/api_manifest.rs::
    // OPERATOR_DESCRIPTORS (Plan 01 regeneration). Each `OPERATOR_DESCRIPTORS[K]`
    // carries `id: OperatorId::new(K)` paired positionally with
    // `entry: &MANIFEST_ENTRIES[K]`. The manifest-agreement `#[test]` in
    // cintx-ops/src/resolver.rs asserts that these constants stay in sync.
    /// `int1e_ecp_cart` operator id (manifest position 26).
    pub const INT1E_ECP_CART: OperatorId = OperatorId::new(26);
    /// `int1e_ecp_sph` operator id (manifest position 27).
    pub const INT1E_ECP_SPH: OperatorId = OperatorId::new(27);
    /// `int1e_ecp_ipnuc_cart` operator id (manifest position 28).
    pub const INT1E_ECP_IPNUC_CART: OperatorId = OperatorId::new(28);
    /// `int1e_ecp_ipnuc_sph` operator id (manifest position 29).
    pub const INT1E_ECP_IPNUC_SPH: OperatorId = OperatorId::new(29);

    /// Returns `true` when `self` identifies one of the four ECP operators
    /// (`int1e_ecp_{cart,sph,ipnuc_cart,ipnuc_sph}`). Used by the safe-API
    /// preflight in `cintx-rs::SessionRequest::query_workspace` to gate
    /// `FacadeError::MissingEcpBasis`.
    pub const fn is_ecp(self) -> bool {
        matches!(self.0, 26 | 27 | 28 | 29)
    }
}

impl From<u32> for OperatorId {
    fn from(value: u32) -> Self {
        OperatorId(value)
    }
}

impl fmt::Display for OperatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op_{:08x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::OperatorId;

    #[test]
    fn is_ecp_returns_true_for_all_four_ecp_ids() {
        assert!(OperatorId::INT1E_ECP_CART.is_ecp());
        assert!(OperatorId::INT1E_ECP_SPH.is_ecp());
        assert!(OperatorId::INT1E_ECP_IPNUC_CART.is_ecp());
        assert!(OperatorId::INT1E_ECP_IPNUC_SPH.is_ecp());
    }

    #[test]
    fn is_ecp_returns_false_for_non_ecp_ids() {
        // int1e_ovlp_cart
        assert!(!OperatorId::new(0).is_ecp());
        // int1e_kin_sph
        assert!(!OperatorId::new(4).is_ecp());
        // int4c1e_cart — preservation invariant: this must NOT be is_ecp().
        assert!(!OperatorId::new(24).is_ecp());
        // edge: id just below and above the ECP block.
        assert!(!OperatorId::new(25).is_ecp());
        assert!(!OperatorId::new(30).is_ecp());
    }

    #[test]
    fn ecp_constants_match_manifest_positions() {
        // Direct integer equality — these constants must agree with
        // OPERATOR_DESCRIPTORS in crates/cintx-ops/src/generated/api_manifest.rs
        // (verified at runtime by the ecp_operator_ids_match_constants
        // #[test] in cintx-ops/src/resolver.rs).
        assert_eq!(OperatorId::INT1E_ECP_CART.raw(), 26);
        assert_eq!(OperatorId::INT1E_ECP_SPH.raw(), 27);
        assert_eq!(OperatorId::INT1E_ECP_IPNUC_CART.raw(), 28);
        assert_eq!(OperatorId::INT1E_ECP_IPNUC_SPH.raw(), 29);
    }
}
