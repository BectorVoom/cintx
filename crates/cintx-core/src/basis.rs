use crate::atom::Atom;
use crate::error::{CoreError, CoreResult};
use crate::shell::{SHELL_TUPLE_CAPACITY, Shell, ShellTuple};
use smallvec::SmallVec;
use std::sync::Arc;

const BASIS_META_CAPACITY: usize = 32;

/// Precomputed metadata describing shell counts and AO offsets for a basis.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisMeta {
    pub shell_offsets: SmallVec<[usize; BASIS_META_CAPACITY]>,
    pub ao_counts: SmallVec<[usize; BASIS_META_CAPACITY]>,
    pub total_ao: usize,
}

impl BasisMeta {
    fn from_shells(shells: &[Arc<Shell>]) -> Self {
        let mut shell_offsets = SmallVec::new();
        let mut ao_counts = SmallVec::new();
        let mut total_ao = 0;

        for shell in shells {
            shell_offsets.push(total_ao);
            let count = shell.ao_per_shell();
            ao_counts.push(count);
            total_ao += count;
        }

        BasisMeta {
            shell_offsets,
            ao_counts,
            total_ao,
        }
    }

    pub fn shell_offset(&self, index: usize) -> Option<usize> {
        self.shell_offsets.get(index).copied()
    }

    pub fn ao_count(&self, index: usize) -> Option<usize> {
        self.ao_counts.get(index).copied()
    }
}

/// Ownership wrapper for atoms and shells plus cached metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisSet {
    atoms: Arc<[Atom]>,
    shells: Arc<[Arc<Shell>]>,
    meta: BasisMeta,
}

impl BasisSet {
    pub fn try_new(atoms: Arc<[Atom]>, shells: Arc<[Arc<Shell>]>) -> CoreResult<Self> {
        if shells.is_empty() {
            return Err(CoreError::EmptyBasis);
        }

        let atom_count = atoms.len();
        for shell in shells.iter() {
            let atom_index = shell.atom_index as usize;
            if atom_index >= atom_count {
                return Err(CoreError::MissingAtomIndex {
                    index: atom_index,
                    total: atom_count,
                });
            }
        }

        let meta = BasisMeta::from_shells(&shells);
        Ok(BasisSet {
            atoms,
            shells,
            meta,
        })
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    pub fn shells(&self) -> &[Arc<Shell>] {
        &self.shells
    }

    pub fn meta(&self) -> &BasisMeta {
        &self.meta
    }

    pub fn shell_tuple_for_indices<I>(&self, idx: I) -> CoreResult<ShellTuple>
    where
        I: IntoIterator<Item = usize>,
    {
        let mut buffer = Vec::new();
        for index in idx {
            let shell = self
                .shells
                .get(index)
                .ok_or(CoreError::ShellIndexOutOfBounds {
                    index,
                    total: self.shells.len(),
                })?;
            buffer.push(shell.clone());
        }
        ShellTuple::try_from_iter(buffer).map_err(|_| CoreError::ShellTupleArityExceeded {
            limit: SHELL_TUPLE_CAPACITY,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NuclearModel;
    use crate::atom::Atom;
    use crate::operator::Representation;
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_owned().into_boxed_slice())
    }

    #[test]
    fn missing_atom_index_is_rejected() {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let shell = Arc::new(
            Shell::try_new(
                1,
                1,
                1,
                1,
                0,
                Representation::Cart,
                arc_f64(&[1.0]),
                arc_f64(&[1.0]),
            )
            .unwrap(),
        );
        let shells = Arc::from(vec![shell].into_boxed_slice());

        let err = BasisSet::try_new(atoms, shells).unwrap_err();
        assert!(matches!(
            err,
            CoreError::MissingAtomIndex { index: 1, total: 1 }
        ));
    }
}
