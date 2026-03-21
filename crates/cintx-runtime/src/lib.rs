//! Runtime planning and workspace governance for cintx.

pub mod options;
pub mod planner;
pub mod validator;
pub mod workspace;

pub use options::ExecutionOptions;
pub use planner::{ExecutionPlan, ExecutionStats, evaluate, query_workspace};
pub use validator::{ValidatedShellTuple, validate_dims, validate_shell_tuple};
pub use workspace::{
    ChunkInfo, ChunkPlan, ChunkPlanner, FallibleBuffer, HostWorkspaceAllocator, WorkspaceAllocator,
    WorkspaceQuery, WorkspaceRequest,
};
