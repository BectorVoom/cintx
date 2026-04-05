//! Runtime planning and workspace governance for cintx.

pub mod dispatch;
pub mod metrics;
pub mod options;
pub mod planner;
pub mod scheduler;
pub mod validator;
pub mod workspace;

pub use dispatch::BackendExecutor;
pub use dispatch::DispatchDecision;
pub use dispatch::DispatchFamily;
pub use dispatch::ExecutionIo;
pub use dispatch::OutputOwnership;
pub use dispatch::WorkspaceBytes;
pub use metrics::ExecutionStats;
pub use options::{BackendCapabilityToken, BackendIntent, BackendKind, ExecutionOptions};
pub use planner::{ExecutionPlan, OperatorEnvParams, OutputLayoutMetadata, evaluate, query_workspace};
pub use scheduler::schedule_chunks;
pub use validator::{ValidatedShellTuple, validate_dims, validate_shell_tuple};
pub use workspace::{
    ChunkInfo, ChunkPlan, ChunkPlanner, FallibleBuffer, HostWorkspaceAllocator, WorkspaceAllocator,
    WorkspaceQuery, WorkspaceRequest,
};
