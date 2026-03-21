use std::sync::Arc;

/// The units used for the environment vector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EnvUnits {
    /// Internally stored in Bohr.
    Bohr,
    /// Provided in Angstrom and converted as needed.
    Angstrom,
}

impl Default for EnvUnits {
    fn default() -> Self {
        EnvUnits::Bohr
    }
}

#[derive(Debug, thiserror::Error)]
#[error("env index {requested} is out of bounds for {available} entries")]
pub struct EnvBoundsError {
    requested: usize,
    available: usize,
}

/// Shared environment parameters backed by an Arc slice.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvParams {
    values: Arc<[f64]>,
    units: Option<EnvUnits>,
}

impl EnvParams {
    pub fn new(values: Arc<[f64]>, units: Option<EnvUnits>) -> Self {
        EnvParams { values, units }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn units(&self) -> Option<EnvUnits> {
        self.units
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    pub fn get_checked(&self, index: usize) -> Result<f64, EnvBoundsError> {
        self.values
            .get(index)
            .copied()
            .ok_or(EnvBoundsError {
                requested: index,
                available: self.values.len(),
            })
    }
}
