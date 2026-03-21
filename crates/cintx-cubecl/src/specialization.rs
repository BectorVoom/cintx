use cintx_core::{Representation, Shell};
use cintx_runtime::ExecutionPlan;
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComponentRank {
    raw: &'static str,
    dims: SmallVec<[usize; 4]>,
}

impl ComponentRank {
    pub fn from_manifest(raw: &'static str) -> Self {
        let mut dims = SmallVec::new();
        for token in raw.split(|ch: char| !ch.is_ascii_digit()) {
            if token.is_empty() {
                continue;
            }
            if let Ok(value) = token.parse::<usize>() {
                dims.push(value.max(1));
            }
        }

        if dims.is_empty() {
            dims.push(1);
        }

        Self { raw, dims }
    }

    pub fn raw(&self) -> &'static str {
        self.raw
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpecializationKey {
    canonical_family: &'static str,
    representation: Representation,
    component_rank: ComponentRank,
    shell_angular_momentum: SmallVec<[u8; 4]>,
}

impl SpecializationKey {
    pub fn from_plan(plan: &ExecutionPlan<'_>) -> Self {
        let shell_angular_momentum = plan
            .shells
            .as_slice()
            .iter()
            .map(|shell| shell.ang_momentum)
            .collect();

        Self {
            canonical_family: plan.descriptor.entry.canonical_family,
            representation: plan.representation,
            component_rank: ComponentRank::from_manifest(plan.descriptor.entry.component_rank),
            shell_angular_momentum,
        }
    }

    pub fn canonical_family(&self) -> &'static str {
        self.canonical_family
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn component_rank(&self) -> &ComponentRank {
        &self.component_rank
    }

    pub fn shell_angular_momentum(&self) -> &[u8] {
        &self.shell_angular_momentum
    }
}

pub(crate) fn hash_shell_tuple(shells: &[Arc<Shell>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut state = std::collections::hash_map::DefaultHasher::new();
    for shell in shells {
        shell.atom_index.hash(&mut state);
        shell.ang_momentum.hash(&mut state);
        shell.nprim.hash(&mut state);
        shell.nctr.hash(&mut state);
        shell.kappa.hash(&mut state);
        shell.representation.hash(&mut state);
    }
    state.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Shell, ShellTuple};
    use cintx_runtime::{query_workspace, ExecutionOptions};

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let shell_a = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[1.0]), arc_f64(&[1.0, 0.5])).unwrap(),
        );
        let shell_b =
            Arc::new(Shell::try_new(0, 2, 1, 1, 0, rep, arc_f64(&[0.8]), arc_f64(&[0.7])).unwrap());

        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice()),
        )
        .unwrap();
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        (basis, shells)
    }

    #[test]
    fn component_rank_defaults_to_scalar() {
        let rank = ComponentRank::from_manifest("");
        assert_eq!(rank.raw(), "");
        assert_eq!(rank.dims(), &[1]);
    }

    #[test]
    fn specialization_key_uses_canonical_family_and_shell_tuple() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .expect("workspace query should succeed");
        let plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .expect("plan should build");

        let key = SpecializationKey::from_plan(&plan);
        assert_eq!(key.canonical_family(), "1e");
        assert_eq!(key.representation(), Representation::Cart);
        assert_eq!(key.component_rank().dims(), &[1]);
        assert_eq!(key.shell_angular_momentum(), &[1, 2]);
    }

    #[test]
    fn shell_tuple_hash_changes_with_angular_momentum() {
        let shell_a = Arc::new(
            Shell::try_new(
                0,
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
        let shell_b = Arc::new(
            Shell::try_new(
                0,
                2,
                1,
                1,
                0,
                Representation::Cart,
                arc_f64(&[1.0]),
                arc_f64(&[1.0]),
            )
            .unwrap(),
        );
        let shell_c = Arc::new(
            Shell::try_new(
                0,
                3,
                1,
                1,
                0,
                Representation::Cart,
                arc_f64(&[1.0]),
                arc_f64(&[1.0]),
            )
            .unwrap(),
        );

        let left = hash_shell_tuple(&[shell_a.clone(), shell_b]);
        let right = hash_shell_tuple(&[shell_a, shell_c]);
        assert_ne!(left, right);
    }
}
