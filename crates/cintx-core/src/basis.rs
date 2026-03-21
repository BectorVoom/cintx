use crate::atom::Atom;
use crate::shell::Shell;
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
            let count = shell.nctr as usize;
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
    pub fn new(atoms: Arc<[Atom]>, shells: Arc<[Arc<Shell>]>) -> Self {
        let meta = BasisMeta::from_shells(&shells);
        BasisSet { atoms, shells, meta }
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
}
