//! Stable prelude re-exports for the safe facade.

pub use crate::api::EvaluationStats;
pub use crate::api::IntegralTensor;
pub use crate::api::SessionQuery;
pub use crate::api::SessionRequest;
pub use crate::api::TypedEvaluationOutput;
pub use crate::api::WorkspaceChunk;
pub use crate::api::WorkspaceExecutionToken;
pub use crate::api::WorkspacePlan;
pub use crate::api::unsupported_unstable_request;
pub use crate::builder::SessionBuilder;
pub use crate::error::{FacadeError, FacadeErrorKind};
pub use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple};
pub use cintx_runtime::ExecutionOptions;

#[cfg(feature = "unstable-source-api")]
pub use crate::api::unstable;
