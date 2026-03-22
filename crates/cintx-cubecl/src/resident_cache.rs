use crate::specialization::hash_shell_tuple;
use cintx_core::{BasisSet, Representation};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResidentCacheKey {
    pub basis_hash: u64,
    pub representation: Representation,
    pub device_profile: Arc<str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentMetadata {
    pub shell_count: usize,
    pub total_ao: usize,
    pub shell_offsets: SmallVec<[usize; 16]>,
    pub ao_counts: SmallVec<[usize; 16]>,
}

impl ResidentMetadata {
    fn from_basis(basis: &BasisSet) -> Self {
        Self {
            shell_count: basis.shells().len(),
            total_ao: basis.meta().total_ao,
            shell_offsets: basis.meta().shell_offsets.iter().copied().collect(),
            ao_counts: basis.meta().ao_counts.iter().copied().collect(),
        }
    }
}

#[derive(Debug, Default)]
pub struct DeviceResidentCache {
    entries: RwLock<HashMap<ResidentCacheKey, Arc<ResidentMetadata>>>,
}

pub type ResidentCache = DeviceResidentCache;

impl DeviceResidentCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn basis_hash(basis: &BasisSet) -> u64 {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        basis.atoms().len().hash(&mut state);
        basis.shells().len().hash(&mut state);
        basis.meta().total_ao.hash(&mut state);
        hash_shell_tuple(basis.shells()).hash(&mut state);
        state.finish()
    }

    pub fn key_for(
        &self,
        device_profile: impl Into<Arc<str>>,
        basis: &BasisSet,
        representation: Representation,
    ) -> ResidentCacheKey {
        ResidentCacheKey {
            basis_hash: Self::basis_hash(basis),
            representation,
            device_profile: device_profile.into(),
        }
    }

    pub fn resident_metadata(
        &self,
        device_profile: impl Into<Arc<str>>,
        basis: &BasisSet,
        representation: Representation,
    ) -> Arc<ResidentMetadata> {
        let key = self.key_for(device_profile, basis, representation);
        if let Some(existing) = self
            .entries
            .read()
            .expect("resident cache poisoned")
            .get(&key)
        {
            return Arc::clone(existing);
        }

        let metadata = Arc::new(ResidentMetadata::from_basis(basis));
        let mut entries = self.entries.write().expect("resident cache poisoned");
        Arc::clone(entries.entry(key).or_insert_with(|| Arc::clone(&metadata)))
    }

    pub fn len(&self) -> usize {
        self.entries.read().expect("resident cache poisoned").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, NuclearModel, Shell};

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation) -> BasisSet {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let shell_a = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[1.0]), arc_f64(&[1.0, 0.5])).unwrap(),
        );
        let shell_b =
            Arc::new(Shell::try_new(0, 2, 1, 1, 0, rep, arc_f64(&[0.8]), arc_f64(&[0.7])).unwrap());
        BasisSet::try_new(atoms, Arc::from(vec![shell_a, shell_b].into_boxed_slice())).unwrap()
    }

    #[test]
    fn resident_cache_is_basis_and_device_scoped() {
        let cache = DeviceResidentCache::new();
        let basis = sample_basis(Representation::Cart);

        let left = cache.resident_metadata("cpu", &basis, Representation::Cart);
        let right = cache.resident_metadata("cpu", &basis, Representation::Cart);
        assert!(Arc::ptr_eq(&left, &right));
        assert_eq!(cache.len(), 1);

        let other_device = cache.resident_metadata("wgpu", &basis, Representation::Cart);
        assert!(!Arc::ptr_eq(&left, &other_device));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn resident_cache_key_includes_representation() {
        let cache = DeviceResidentCache::new();
        let basis = sample_basis(Representation::Cart);

        let cart = cache.resident_metadata("cpu", &basis, Representation::Cart);
        let spinor = cache.resident_metadata("cpu", &basis, Representation::Spinor);

        assert!(!Arc::ptr_eq(&cart, &spinor));
        assert_eq!(cache.len(), 2);
    }
}
