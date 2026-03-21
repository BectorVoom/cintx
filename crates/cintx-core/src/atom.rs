use std::fmt;

use crate::error::{CoreError, CoreResult};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_atomic_number_is_rejected() {
        let err = Atom::try_new(0, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap_err();

        assert!(matches!(err, CoreError::InvalidAtomicNumber(0)));
    }

    #[test]
    fn nonfinite_coordinates_are_rejected() {
        let err = Atom::try_new(
            1,
            [f64::INFINITY, 0.0, 0.0],
            NuclearModel::Point,
            None,
            None,
        )
        .unwrap_err();

        if let CoreError::InvalidCoordinate(coords) = err {
            assert!(coords[0].is_infinite());
        } else {
            panic!("expected InvalidCoordinate error");
        }
    }
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
    const MAX_ATOMIC_NUMBER: u16 = 118;

    /// Create a validated atom record.
    pub fn try_new(
        atomic_number: u16,
        coord_bohr: [f64; 3],
        nuclear_model: NuclearModel,
        zeta: Option<f64>,
        fractional_charge: Option<f64>,
    ) -> CoreResult<Self> {
        if atomic_number == 0 || atomic_number > Self::MAX_ATOMIC_NUMBER {
            return Err(CoreError::InvalidAtomicNumber(atomic_number));
        }

        if !coord_bohr.iter().all(|value| value.is_finite()) {
            return Err(CoreError::InvalidCoordinate(coord_bohr));
        }

        if let Some(z) = zeta
            && (!z.is_finite() || z <= 0.0)
        {
            return Err(CoreError::InvalidNuclearDetail);
        }

        if let Some(frac) = fractional_charge
            && (!frac.is_finite() || frac.abs() > 2.0)
        {
            return Err(CoreError::InvalidFractionalCharge(frac));
        }

        Ok(Self {
            atomic_number,
            coord_bohr,
            nuclear_model,
            zeta,
            fractional_charge,
        })
    }
}
