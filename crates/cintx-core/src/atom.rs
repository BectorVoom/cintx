use std::fmt;

/// Nuclear model governing how the shell feels the nucleus.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NuclearModel {
    /// Point nucleus (default in libcint).
    Point,
    /// Finite Gaussian nucleus with the supplied `zeta`.
    Gaussian,
    /// Finite spherical nucleus with explicit fractional charge.
    FiniteSpherical,
}

impl fmt::Display for NuclearModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                NuclearModel::Point => "Point",
                NuclearModel::Gaussian => "Gaussian",
                NuclearModel::FiniteSpherical => "FiniteSpherical",
            }
        )
    }
}

/// Immutable atom metadata backed by shared memory for the safe API.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub atomic_number: u16,
    pub coord_bohr: [f64; 3],
    pub nuclear_model: NuclearModel,
    pub zeta: Option<f64>,
    pub fractional_charge: Option<f64>,
}

impl Atom {
    /// Create a new atom record; callers can set optional fields later if needed.
    pub fn new(
        atomic_number: u16,
        coord_bohr: [f64; 3],
        nuclear_model: NuclearModel,
        zeta: Option<f64>,
        fractional_charge: Option<f64>,
    ) -> Self {
        Self {
            atomic_number,
            coord_bohr,
            nuclear_model,
            zeta,
            fractional_charge,
        }
    }
}
