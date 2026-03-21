use crate::resident_cache::DeviceResidentCache;
use crate::specialization::SpecializationKey;
use crate::transfer::TransferPlan;
use cintx_core::cintxRsError;
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionPlan, ExecutionStats, OutputOwnership, WorkspaceBytes,
};

pub const CUBECL_RUNTIME_PROFILE: &str = "cpu";

#[derive(Debug, Default)]
pub struct CubeClExecutor {
    runtime_profile: &'static str,
    resident_cache: DeviceResidentCache,
}

impl CubeClExecutor {
    pub fn new() -> Self {
        Self::with_runtime_profile(CUBECL_RUNTIME_PROFILE)
    }

    pub fn with_runtime_profile(runtime_profile: &'static str) -> Self {
        Self {
            runtime_profile,
            resident_cache: DeviceResidentCache::new(),
        }
    }

    pub fn runtime_profile(&self) -> &'static str {
        self.runtime_profile
    }

    pub fn resident_cache(&self) -> &DeviceResidentCache {
        &self.resident_cache
    }

    fn supports_canonical_family(family: &str) -> bool {
        matches!(family, "1e" | "2e" | "2c2e")
    }

    fn ensure_supported_family(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        let canonical_family = plan.descriptor.entry.canonical_family;
        if canonical_family == "4c1e" {
            return Err(cintxRsError::UnsupportedApi {
                requested: "4c1e remains out of scope for Phase 2 CubeCL executor".to_owned(),
            });
        }

        if !Self::supports_canonical_family(canonical_family) {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "CubeCL executor family {canonical_family} is not enabled in the 1e/2e/2c2e slice"
                ),
            });
        }

        Ok(())
    }
}

impl BackendExecutor for CubeClExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool {
        Self::supports_canonical_family(plan.descriptor.entry.canonical_family)
            && plan
                .descriptor
                .entry
                .supports_representation(plan.representation)
    }

    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError> {
        self.ensure_supported_family(plan)?;
        Ok(WorkspaceBytes(plan.workspace.bytes))
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        self.ensure_supported_family(plan)?;
        io.ensure_output_contract()?;

        if io.backend_output_ownership() != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_executor",
                detail: "backend_output must remain staging-only".to_owned(),
            });
        }
        if io.final_write_ownership() != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_executor",
                detail: "CompatFinalWrite must remain owned by compat layout".to_owned(),
            });
        }

        let _specialization = SpecializationKey::from_plan(plan);
        let _resident = self.resident_cache.resident_metadata(
            self.runtime_profile,
            plan.basis,
            plan.representation,
        );
        let transfer_plan = TransferPlan::from_plan(plan, io.chunk())?;
        transfer_plan.ensure_output_contract()?;
        let transfer = transfer_plan.stage_device_buffers(self.runtime_profile)?;

        // Phase 2 keeps backend output as staging only; compat owns final flat writes.
        for value in io.staging_output().iter_mut() {
            *value = 0.0;
        }

        Ok(ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: transfer.workspace_bytes.max(io.workspace().len()),
            chunk_count: 1,
            planned_batches: io.chunk().work_unit_count,
            transfer_bytes: transfer.transfer_bytes,
            not0: i32::from(!io.staging_output().is_empty()),
            fallback_reason: plan.workspace.fallback_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_runtime::{query_workspace, ExecutionOptions, FallibleBuffer};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation, shell_count: usize) -> BasisSet {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let mut shells = Vec::with_capacity(shell_count);
        for index in 0..shell_count {
            let l = (index % 3 + 1) as u8;
            shells.push(Arc::new(
                Shell::try_new(
                    0,
                    l,
                    1,
                    1,
                    0,
                    rep,
                    arc_f64(&[1.0 - (index as f64 * 0.05)]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            ));
        }

        BasisSet::try_new(atoms, Arc::from(shells.into_boxed_slice())).unwrap()
    }

    fn shell_tuple_for_first_n(basis: &BasisSet, count: usize) -> ShellTuple {
        ShellTuple::try_from_iter(
            basis
                .shells()
                .iter()
                .take(count)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn build_plan(
        basis: &'static BasisSet,
        operator_id: u32,
        rep: Representation,
        arity: usize,
    ) -> ExecutionPlan<'static> {
        let shells = shell_tuple_for_first_n(basis, arity);
        let query = query_workspace(
            OperatorId::new(operator_id),
            rep,
            basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        ExecutionPlan::new(OperatorId::new(operator_id), rep, basis, shells, query).unwrap()
    }

    #[test]
    fn runtime_profile_defaults_to_cpu() {
        let executor = CubeClExecutor::new();
        assert_eq!(executor.runtime_profile(), "cpu");
        assert_eq!(CUBECL_RUNTIME_PROFILE, "cpu");
    }

    #[test]
    fn supports_only_initial_phase2_families() {
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let executor = CubeClExecutor::new();

        let one_e = build_plan(basis, 0, Representation::Cart, 2);
        let two_e = build_plan(basis, 9, Representation::Cart, 4);
        let two_c2e = build_plan(basis, 12, Representation::Cart, 2);
        let three_c1e = build_plan(basis, 15, Representation::Cart, 3);

        assert!(executor.supports(&one_e));
        assert!(executor.supports(&two_e));
        assert!(executor.supports(&two_c2e));
        assert!(!executor.supports(&three_c1e));
    }

    #[test]
    fn unsupported_family_returns_typed_error() {
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let executor = CubeClExecutor::new();
        let plan = build_plan(basis, 15, Representation::Cart, 3);

        let query = plan.workspace.clone();
        let chunk = query.chunks[0].clone();
        let mut staging = vec![0.0; 64];
        let mut workspace =
            FallibleBuffer::try_uninit(query.bytes.max(1), query.alignment).unwrap();
        let mut io = ExecutionIo::new(&chunk, &mut staging, &mut workspace, plan.dispatch).unwrap();

        let err = executor.execute(&plan, &mut io).unwrap_err();
        assert!(matches!(err, cintxRsError::UnsupportedApi { .. }));
    }
}
