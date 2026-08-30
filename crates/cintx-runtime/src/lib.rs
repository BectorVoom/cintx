//! Runtime planning and workspace governance for cintx.

pub mod batch;
pub mod dispatch;
pub mod metrics;
pub mod options;
pub mod planner;
pub mod range_omega;
pub mod scheduler;
pub mod validator;
pub mod workspace;

pub use batch::{
    BatchBucket, BatchChunk, BatchExecutionPlan, BatchItem, BatchItemRequest, KernelClass,
};
pub use dispatch::BackendExecutor;
pub use dispatch::DispatchDecision;
pub use dispatch::DispatchFamily;
pub use dispatch::ExecutionIo;
pub use dispatch::OutputOwnership;
pub use dispatch::WorkspaceBytes;
pub use metrics::ExecutionStats;
pub use options::{BackendCapabilityToken, BackendIntent, BackendKind, ExecutionOptions};
pub use planner::{
    ExecutionPlan, GridsEnvParams, OperatorEnvParams, OutputLayoutMetadata, evaluate,
    query_workspace,
};
pub use range_omega::{
    EXPCUTOFF_SR, PTR_RANGE_OMEGA, SR_DOUBLED_ROOT_MAX_ORDER, derivative_headroom,
    is_range_separated, is_short_range, nrys_roots_for, supports_range_omega,
};
pub use scheduler::schedule_chunks;
pub use validator::{
    ValidatedShellTuple, validate_dims, validate_f12_env_params, validate_grids_env_params,
    validate_range_omega_env_params, validate_range_omega_value, validate_shell_tuple,
};
pub use workspace::{
    ChunkInfo, ChunkPlan, ChunkPlanner, FallibleBuffer, HostWorkspaceAllocator,
    ReusableWorkspaceAllocator, WorkspaceAllocator, WorkspaceQuery, WorkspaceRequest,
};
