use cintx_core::cintxRsError;
use cintx_runtime::{ChunkInfo, ExecutionPlan, OutputOwnership};
use std::mem::size_of;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShellTransferMetadata {
    pub shell_index: usize,
    pub atom_index: u32,
    pub ang_momentum: u8,
    pub ao_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferPlan {
    pub shell_metadata: Vec<ShellTransferMetadata>,
    pub workspace_bytes: usize,
    pub staging_elements: usize,
    pub staging_bytes: usize,
    pub transfer_bytes: usize,
    pub output_ownership: OutputOwnership,
    pub final_write_ownership: OutputOwnership,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferWorkspaceBuffers {
    pub workspace_bytes: usize,
    pub staging_bytes: usize,
    pub transfer_bytes: usize,
}

impl TransferPlan {
    pub fn from_plan(plan: &ExecutionPlan<'_>, chunk: &ChunkInfo) -> Result<Self, cintxRsError> {
        let mut shell_metadata = Vec::new();
        shell_metadata
            .try_reserve_exact(plan.shells.as_slice().len())
            .map_err(|_| cintxRsError::HostAllocationFailed {
                bytes: plan
                    .shells
                    .as_slice()
                    .len()
                    .saturating_mul(size_of::<ShellTransferMetadata>()),
            })?;
        for (shell_index, shell) in plan.shells.as_slice().iter().enumerate() {
            shell_metadata.push(ShellTransferMetadata {
                shell_index,
                atom_index: shell.atom_index,
                ang_momentum: shell.ang_momentum,
                ao_count: shell.ao_per_shell(),
            });
        }

        let staging_elements = chunk_staging_elements(plan, chunk)?;
        let staging_bytes = staging_elements
            .checked_mul(size_of::<f64>())
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
        let metadata_bytes = shell_metadata
            .len()
            .checked_mul(size_of::<ShellTransferMetadata>())
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
        let transfer_bytes = chunk
            .bytes
            .checked_add(staging_bytes)
            .and_then(|value| value.checked_add(metadata_bytes))
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;

        Ok(Self {
            shell_metadata,
            workspace_bytes: chunk.bytes,
            staging_elements,
            staging_bytes,
            transfer_bytes,
            output_ownership: plan.dispatch.backend_output,
            final_write_ownership: plan.dispatch.final_write,
        })
    }

    pub fn ensure_output_contract(&self) -> Result<(), cintxRsError> {
        if self.output_ownership != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "transfer",
                detail: "CubeCL output ownership must remain BackendStagingOnly".to_owned(),
            });
        }
        if self.final_write_ownership != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "transfer",
                detail: "CompatFinalWrite ownership must stay outside transfer planning".to_owned(),
            });
        }
        Ok(())
    }

    pub fn stage_output_buffer(&self) -> Result<Vec<f64>, cintxRsError> {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(self.staging_elements)
            .map_err(|_| cintxRsError::HostAllocationFailed {
                bytes: self.staging_bytes,
            })?;
        buffer.resize(self.staging_elements, 0.0);
        Ok(buffer)
    }

    pub fn stage_device_buffers(
        &self,
        device_profile: &str,
    ) -> Result<TransferWorkspaceBuffers, cintxRsError> {
        let mut staging_workspace_probe = Vec::<u8>::new();
        staging_workspace_probe
            .try_reserve_exact(self.workspace_bytes.max(1))
            .map_err(|_| cintxRsError::DeviceOutOfMemory {
                bytes: self.workspace_bytes,
                device: device_profile.to_owned(),
            })?;

        Ok(TransferWorkspaceBuffers {
            workspace_bytes: self.workspace_bytes,
            staging_bytes: self.staging_bytes,
            transfer_bytes: self.transfer_bytes,
        })
    }
}

fn chunk_staging_elements(
    plan: &ExecutionPlan<'_>,
    chunk: &ChunkInfo,
) -> Result<usize, cintxRsError> {
    let total_units = plan.workspace.work_units.max(1);
    let start = chunk.work_unit_start.min(total_units);
    let end = chunk
        .work_unit_start
        .checked_add(chunk.work_unit_count)
        .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?
        .min(total_units);

    let prefix = plan.output_layout.staging_elements.saturating_mul(start) / total_units;
    let suffix = plan.output_layout.staging_elements.saturating_mul(end) / total_units;
    Ok(suffix.saturating_sub(prefix).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_runtime::{query_workspace, ExecutionOptions};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_plan() -> ExecutionPlan<'static> {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
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
                arc_f64(&[0.8]),
                arc_f64(&[0.7]),
            )
            .unwrap(),
        );
        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice()),
        )
        .unwrap();
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let basis = Box::leak(Box::new(basis));
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells,
            query,
        )
        .unwrap()
    }

    #[test]
    fn transfer_plan_preserves_staging_only_contract() {
        let plan = sample_plan();
        let chunk = &plan.workspace.chunks[0];
        let transfer = TransferPlan::from_plan(&plan, chunk).unwrap();
        transfer.ensure_output_contract().unwrap();
        assert_eq!(
            transfer.output_ownership,
            OutputOwnership::BackendStagingOnly
        );
        assert_eq!(
            transfer.final_write_ownership,
            OutputOwnership::CompatFinalWrite
        );
    }
}
