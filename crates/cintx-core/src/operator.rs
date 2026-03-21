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
