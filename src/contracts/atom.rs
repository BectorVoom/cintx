use super::{ContractError, ContractResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    atomic_number: u8,
    coordinates: [f64; 3],
}

impl Atom {
    pub fn new(atomic_number: u8, coordinates: [f64; 3]) -> ContractResult<Self> {
        if atomic_number == 0 {
            return Err(ContractError::InvalidInput {
                field: "atomic_number",
                reason: "must be greater than zero".to_string(),
            });
        }

        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidInput {
                field: "coordinates",
                reason: "all coordinate values must be finite".to_string(),
            });
        }

        Ok(Self {
            atomic_number,
            coordinates,
        })
    }

    pub fn atomic_number(&self) -> u8 {
        self.atomic_number
    }

    pub fn coordinates(&self) -> [f64; 3] {
        self.coordinates
    }
}
