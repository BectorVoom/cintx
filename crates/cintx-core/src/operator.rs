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
