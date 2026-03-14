use super::{ContractError, ContractResult};

const MAX_ANGULAR_MOMENTUM: u8 = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct ShellPrimitive {
    exponent: f64,
    coefficient: f64,
}

impl ShellPrimitive {
    pub fn new(exponent: f64, coefficient: f64) -> ContractResult<Self> {
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(ContractError::InvalidInput {
                field: "exponent",
                reason: "must be a finite value greater than zero".to_string(),
            });
        }

        if !coefficient.is_finite() {
            return Err(ContractError::InvalidInput {
                field: "coefficient",
                reason: "must be a finite value".to_string(),
            });
        }

        Ok(Self {
            exponent,
            coefficient,
        })
    }

    pub fn exponent(&self) -> f64 {
        self.exponent
    }

    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    center_index: usize,
    angular_momentum: u8,
    primitives: Vec<ShellPrimitive>,
}

impl Shell {
    pub fn new(
        center_index: usize,
        angular_momentum: u8,
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
    ) -> ContractResult<Self> {
        if exponents.is_empty() {
            return Err(ContractError::InvalidInput {
                field: "exponents",
                reason: "must contain at least one primitive".to_string(),
            });
        }

        if exponents.len() != coefficients.len() {
            return Err(ContractError::InvalidLayout {
                item: "primitive coefficients",
                expected: exponents.len(),
                got: coefficients.len(),
            });
        }

        if angular_momentum > MAX_ANGULAR_MOMENTUM {
            return Err(ContractError::Unsupported {
                field: "angular_momentum",
                value: "value exceeds supported metadata range",
            });
        }

        let mut primitives = Vec::with_capacity(exponents.len());
        for (exponent, coefficient) in exponents.into_iter().zip(coefficients.into_iter()) {
            primitives.push(ShellPrimitive::new(exponent, coefficient)?);
        }

        if primitives
            .iter()
            .all(|primitive| primitive.coefficient() == 0.0)
        {
            return Err(ContractError::InvalidInput {
                field: "coefficients",
                reason: "at least one primitive coefficient must be non-zero".to_string(),
            });
        }

        Ok(Self {
            center_index,
            angular_momentum,
            primitives,
        })
    }

    pub fn center_index(&self) -> usize {
        self.center_index
    }

    pub fn angular_momentum(&self) -> u8 {
        self.angular_momentum
    }

    pub fn primitives(&self) -> &[ShellPrimitive] {
        &self.primitives
    }
}
