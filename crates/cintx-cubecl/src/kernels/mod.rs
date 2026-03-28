pub mod center_2c2e;
pub mod center_3c1e;
pub mod center_3c2e;
#[cfg(feature = "with-4c1e")]
pub mod center_4c1e;
pub mod one_electron;
pub mod two_electron;

use crate::specialization::SpecializationKey;
use crate::transfer::TransferPlan;
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats};

pub type FamilyLaunchFn = fn(
    &ExecutionPlan<'_>,
    &SpecializationKey,
    &TransferPlan,
) -> Result<ExecutionStats, cintxRsError>;

#[cfg(not(feature = "with-4c1e"))]
const UNSUPPORTED_FOLLOW_ON_FAMILIES: &[&str] = &["center_4c1e"];
#[cfg(feature = "with-4c1e")]
const UNSUPPORTED_FOLLOW_ON_FAMILIES: &[&str] = &[];

fn resolve_family_name(canonical_family: &str) -> Option<FamilyLaunchFn> {
    match canonical_family {
        "1e" => Some(one_electron::launch_one_electron as FamilyLaunchFn),
        "2e" => Some(two_electron::launch_two_electron as FamilyLaunchFn),
        "2c2e" => Some(center_2c2e::launch_center_2c2e as FamilyLaunchFn),
        "3c1e" => Some(center_3c1e::launch_center_3c1e as FamilyLaunchFn),
        "3c2e" => Some(center_3c2e::launch_center_3c2e as FamilyLaunchFn),
        #[cfg(feature = "with-4c1e")]
        "4c1e" => Some(center_4c1e::launch_center_4c1e as FamilyLaunchFn),
        _ => None,
    }
}

pub fn supports_canonical_family(canonical_family: &str) -> bool {
    match canonical_family {
        "1e" | "2e" | "2c2e" | "3c1e" | "3c2e" => true,
        "4c1e" => cfg!(feature = "with-4c1e"),
        _ => false,
    }
}

pub fn unresolved_families() -> &'static [&'static str] {
    UNSUPPORTED_FOLLOW_ON_FAMILIES
}

pub fn resolve_family(plan: &ExecutionPlan<'_>) -> Result<FamilyLaunchFn, cintxRsError> {
    if !plan
        .descriptor
        .entry
        .supports_representation(plan.representation)
    {
        return Err(cintxRsError::UnsupportedRepresentation {
            operator: format!(
                "{}/{}",
                plan.descriptor.entry.canonical_family,
                plan.descriptor.operator_name()
            ),
            representation: plan.representation,
        });
    }

    let canonical_family = plan.descriptor.entry.canonical_family;
    resolve_family_name(canonical_family).ok_or_else(|| cintxRsError::UnsupportedApi {
        requested: format!(
            "CubeCL family registry does not include canonical_family={canonical_family}"
        ),
    })
}

pub fn launch_family(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    transfer_plan: &TransferPlan,
) -> Result<ExecutionStats, cintxRsError> {
    let launch = resolve_family(plan)?;
    launch(plan, specialization, transfer_plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, query_workspace};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation, shell_count: usize) -> BasisSet {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let mut shells = Vec::with_capacity(shell_count);
        for index in 0..shell_count {
            shells.push(Arc::new(
                Shell::try_new(
                    0,
                    (index % 3 + 1) as u8,
                    1,
                    1,
                    0,
                    rep,
                    arc_f64(&[1.0]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            ));
        }
        BasisSet::try_new(atoms, Arc::from(shells.into_boxed_slice())).unwrap()
    }

    fn build_plan(
        basis: &'static BasisSet,
        operator_id: u32,
        representation: Representation,
        arity: usize,
    ) -> ExecutionPlan<'static> {
        let shells = ShellTuple::try_from_iter(
            basis
                .shells()
                .iter()
                .take(arity)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let query = query_workspace(
            OperatorId::new(operator_id),
            representation,
            basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        ExecutionPlan::new(
            OperatorId::new(operator_id),
            representation,
            basis,
            shells,
            query,
        )
        .unwrap()
    }

    #[test]
    fn family_registry_resolves_base_slice() {
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let one_e = build_plan(basis, 0, Representation::Cart, 2);
        let two_e = build_plan(basis, 9, Representation::Cart, 4);
        let two_c2e = build_plan(basis, 12, Representation::Cart, 2);
        let three_c1e = build_plan(basis, 15, Representation::Cart, 3);
        let three_c2e = build_plan(basis, 17, Representation::Cart, 3);

        assert!(resolve_family(&one_e).is_ok());
        assert!(resolve_family(&two_e).is_ok());
        assert!(resolve_family(&two_c2e).is_ok());
        assert!(resolve_family(&three_c1e).is_ok());
        assert!(resolve_family(&three_c2e).is_ok());
    }

    #[test]
    #[cfg(not(feature = "with-4c1e"))]
    fn family_registry_rejects_unsupported_4c1e_when_feature_disabled() {
        let op_4c1e = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("4c1e descriptor should exist")
            .entry
            .canonical_family;
        assert_eq!(op_4c1e, "4c1e");
        assert!(!supports_canonical_family(op_4c1e));
        assert!(resolve_family_name(op_4c1e).is_none());
        assert_eq!(unresolved_families(), &["center_4c1e"]);
    }

    #[test]
    fn supports_function_tracks_registry() {
        assert!(supports_canonical_family("1e"));
        assert!(supports_canonical_family("2e"));
        assert!(supports_canonical_family("2c2e"));
        assert!(supports_canonical_family("3c1e"));
        assert!(supports_canonical_family("3c2e"));
        #[cfg(feature = "with-4c1e")]
        assert!(supports_canonical_family("4c1e"));
        #[cfg(not(feature = "with-4c1e"))]
        assert!(!supports_canonical_family("4c1e"));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn family_registry_enables_4c1e_when_feature_enabled() {
        let op_4c1e = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("4c1e descriptor should exist")
            .entry
            .canonical_family;
        assert_eq!(op_4c1e, "4c1e");
        assert!(supports_canonical_family(op_4c1e));
        assert!(resolve_family_name(op_4c1e).is_some());
        assert!(unresolved_families().is_empty());
    }
}
