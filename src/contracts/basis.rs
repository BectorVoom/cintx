use super::{Atom, ContractError, ContractResult, Shell};

#[derive(Debug, Clone, PartialEq)]
pub struct BasisSet {
    atoms: Vec<Atom>,
    shells: Vec<Shell>,
}

impl BasisSet {
    pub fn new(atoms: Vec<Atom>, shells: Vec<Shell>) -> ContractResult<Self> {
        if atoms.is_empty() {
            return Err(ContractError::InvalidInput {
                field: "atoms",
                reason: "basis requires at least one atom".to_string(),
            });
        }

        if shells.is_empty() {
            return Err(ContractError::InvalidInput {
                field: "shells",
                reason: "basis requires at least one shell".to_string(),
            });
        }

        for shell in &shells {
            if shell.center_index() >= atoms.len() {
                return Err(ContractError::OutOfBounds {
                    field: "shell.center_index",
                    index: shell.center_index(),
                    len: atoms.len(),
                    collection: "atoms",
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
