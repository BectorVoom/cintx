use cintx_core::{BasisSet, EcpChannel, Representation};
use smallvec::SmallVec;
use std::collections::HashMap;
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
        // Device-resident values must never alias on structural similarity
        // alone. Hash all result-affecting fields as exact IEEE bytes.
        let mut state = StableBasisHasher::new();
        state.usize(basis.atoms().len());
        for atom in basis.atoms() {
            state.u16(atom.atomic_number);
            for coordinate in atom.coord_bohr {
                state.f64(coordinate);
            }
            state.u8(atom.nuclear_model as u8);
            state.option_f64(atom.zeta);
            state.option_f64(atom.fractional_charge);
        }

        state.usize(basis.shells().len());
        for shell in basis.shells() {
            state.u32(shell.atom_index);
            state.u8(shell.ang_momentum);
            state.u16(shell.nprim);
            state.u16(shell.nctr);
            state.i16(shell.kappa);
            state.u8(shell.representation as u8);
            state.f64_slice(&shell.exponents);
            state.f64_slice(&shell.coefficients);
        }

        state.usize(basis.ecp_shells().len());
        for shell in basis.ecp_shells() {
            state.u32(shell.atom_index);
            match shell.channel {
                EcpChannel::Local => state.u8(0),
                EcpChannel::Projected(l) => {
                    state.u8(1);
                    state.u8(l);
                }
            }
            state.i16(shell.radial_power);
            state.u16(shell.nprim);
            state.u16(shell.nctr);
            state.i16(shell.so_type);
            state.f64_slice(&shell.exponents);
            state.f64_slice(&shell.coefficients);
        }
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

/// Fixed byte-wise FNV-1a state. This is deliberately stable across processes,
/// unlike `DefaultHasher`, because cache identity is an execution contract.
struct StableBasisHasher(u64);

impl StableBasisHasher {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
    fn u8(&mut self, value: u8) {
        self.bytes(&value.to_le_bytes());
    }
    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }
    fn i16(&mut self, value: i16) {
        self.bytes(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) {
        self.bytes(&(value as u64).to_le_bytes());
    }
    fn f64(&mut self, value: f64) {
        self.bytes(&value.to_bits().to_le_bytes());
    }
    fn option_f64(&mut self, value: Option<f64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.f64(value);
            }
            None => self.u8(0),
        }
    }
    fn f64_slice(&mut self, values: &[f64]) {
        self.usize(values.len());
        for value in values {
            self.f64(*value);
        }
    }
    const fn finish(self) -> u64 {
        self.0
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

    #[test]
    fn resident_cache_hash_includes_numeric_basis_content() {
        let left = sample_basis(Representation::Cart);
        let atom = Atom::try_new(1, [0.125, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let shell_a = Arc::new(
            Shell::try_new(
                0,
                1,
                1,
                2,
                0,
                Representation::Cart,
                arc_f64(&[1.0]),
                arc_f64(&[1.0, 0.25]),
            )
            .unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(
                0,
                2,
                1,
                1,
                0,
                Representation::Cart,
                arc_f64(&[0.8]),
                arc_f64(&[0.7]),
            )
            .unwrap(),
        );
        let right = BasisSet::try_new(
            Arc::from(vec![atom].into_boxed_slice()),
            Arc::from(vec![shell_a, shell_b].into_boxed_slice()),
        )
        .unwrap();

        assert_ne!(
            DeviceResidentCache::basis_hash(&left),
            DeviceResidentCache::basis_hash(&right)
        );
    }
}
