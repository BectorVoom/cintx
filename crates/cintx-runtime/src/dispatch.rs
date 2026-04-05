use crate::metrics::ExecutionStats;
use crate::planner::ExecutionPlan;
use crate::workspace::{ChunkInfo, FallibleBuffer};
use cintx_core::cintxRsError;

/// Runtime-level dispatch families supported by the shared planner path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchFamily {
    OneElectron,
    TwoElectron,
    Center2c2e,
    Center3c1e,
    Center3c2e,
    Center4c1e,
    /// Unstable-source families (gated by `unstable-source-api` feature).
    /// All five unstable families share this variant; operator_name
    /// disambiguates within the kernel dispatch.
    Origi,
    Grids,
    Breit,
    Origk,
    Ssc,
}

impl DispatchFamily {
    pub fn from_manifest_family(family: &str) -> Result<Self, cintxRsError> {
        match family {
            "1e" => Ok(Self::OneElectron),
            "2e" => Ok(Self::TwoElectron),
            "2c2e" => Ok(Self::Center2c2e),
            "3c1e" => Ok(Self::Center3c1e),
            "3c2e" => Ok(Self::Center3c2e),
            "4c1e" => Ok(Self::Center4c1e),
            // Unstable-source families (Phase 14).
            // family_name uses "unstable::source::{name}" prefix; canonical_family is the short name.
            // DispatchDecision is built from descriptor.family() = entry.family_name.
            "unstable::source::origi" => Ok(Self::Origi),
            "unstable::source::grids" => Ok(Self::Grids),
            "unstable::source::breit" => Ok(Self::Breit),
            "unstable::source::origk" => Ok(Self::Origk),
            "unstable::source::ssc" => Ok(Self::Ssc),
            _ => Err(cintxRsError::UnsupportedApi {
                requested: format!("unsupported dispatch family {family}"),
            }),
        }
    }
}

/// Declares who owns caller-visible output writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputOwnership {
    BackendStagingOnly,
    CompatFinalWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceBytes(pub usize);

impl WorkspaceBytes {
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Planner-owned dispatch metadata for backend execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchDecision {
    pub family: DispatchFamily,
    pub backend_output: OutputOwnership,
    pub final_write: OutputOwnership,
}

impl DispatchDecision {
    pub fn from_manifest_family(family: &str) -> Result<Self, cintxRsError> {
        Ok(Self {
            family: DispatchFamily::from_manifest_family(family)?,
            backend_output: OutputOwnership::BackendStagingOnly,
            final_write: OutputOwnership::CompatFinalWrite,
        })
    }

    pub fn ensure_output_contract(self) -> Result<(), cintxRsError> {
        if self.backend_output != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "dispatch",
                detail: "backend output ownership must stay staging-only".to_owned(),
            });
        }
        if self.final_write != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "dispatch",
                detail: "caller-visible flat writes belong to compat layout".to_owned(),
            });
        }
        Ok(())
    }
}

/// Runtime-owned IO contract passed from the planner to a backend executor.
///
/// Backends may fill staging output and workspace metadata but never own
/// caller-visible flat writes.
#[derive(Debug)]
pub struct ExecutionIo<'a> {
    chunk: &'a ChunkInfo,
    staging_output: &'a mut [f64],
    workspace: &'a mut FallibleBuffer<u8>,
    backend_output: OutputOwnership,
    final_write: OutputOwnership,
    transfer_bytes: usize,
    not0: i32,
}

impl<'a> ExecutionIo<'a> {
    pub fn new(
        chunk: &'a ChunkInfo,
        staging_output: &'a mut [f64],
        workspace: &'a mut FallibleBuffer<u8>,
        dispatch: DispatchDecision,
    ) -> Result<Self, cintxRsError> {
        dispatch.ensure_output_contract()?;
        Ok(Self {
            chunk,
            staging_output,
            workspace,
            backend_output: dispatch.backend_output,
            final_write: dispatch.final_write,
            transfer_bytes: 0,
            not0: 0,
        })
    }

    pub fn chunk(&self) -> &ChunkInfo {
        self.chunk
    }

    pub fn staging_output(&mut self) -> &mut [f64] {
        self.staging_output
    }

    pub fn workspace(&mut self) -> &mut FallibleBuffer<u8> {
        self.workspace
    }

    pub fn backend_output_ownership(&self) -> OutputOwnership {
        self.backend_output
    }

    pub fn final_write_ownership(&self) -> OutputOwnership {
        self.final_write
    }

    pub fn transfer_bytes(&self) -> usize {
        self.transfer_bytes
    }

    pub fn not0(&self) -> i32 {
        self.not0
    }

    pub fn record_transfer_bytes(&mut self, bytes: usize) {
        self.transfer_bytes = self.transfer_bytes.saturating_add(bytes);
    }

    pub fn record_not0(&mut self, not0: i32) {
        self.not0 = self.not0.saturating_add(not0.max(0));
    }

    pub fn ensure_output_contract(&self) -> Result<(), cintxRsError> {
        if self.backend_output != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "execution_io",
                detail: "backend attempted to claim caller-visible output ownership".to_owned(),
            });
        }
        if self.final_write != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "execution_io",
                detail: "compat final-write contract changed during execution".to_owned(),
            });
        }
        Ok(())
    }
}

/// Backend-neutral runtime contract.
pub trait BackendExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool;
    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError>;
    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError>;
}
