#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Representation {
    Cartesian,
    Spherical,
    Spinor,
}

impl Representation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cartesian => "cartesian",
            Self::Spherical => "spherical",
            Self::Spinor => "spinor",
        }
    }
}
