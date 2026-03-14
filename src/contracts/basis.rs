use super::{Atom, ContractResult, Shell};
use crate::errors::LibcintRsError;

#[derive(Debug, Clone, PartialEq)]
pub struct BasisSet {
    atoms: Vec<Atom>,
    shells: Vec<Shell>,
}

impl BasisSet {
    pub fn new(atoms: Vec<Atom>, shells: Vec<Shell>) -> ContractResult<Self> {
        if atoms.is_empty() {
            return Err(LibcintRsError::InvalidInput {
                field: "atoms",
                reason: "basis requires at least one atom".to_string(),
            });
        }

        if shells.is_empty() {
            return Err(LibcintRsError::InvalidInput {
                field: "shells",
                reason: "basis requires at least one shell".to_string(),
            });
        }

        for shell in &shells {
            if shell.center_index() >= atoms.len() {
                return Err(LibcintRsError::InvalidInput {
                    field: "shell.center_index",
                    reason: format!(
                        "index {} is out of bounds for {} atoms",
                        shell.center_index(),
                        atoms.len()
                    ),
                });
            }
        }

        Ok(Self { atoms, shells })
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }
}
