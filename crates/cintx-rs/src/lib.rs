//! Facade crate for the cintx workspace.

pub mod api;
pub mod builder;
pub mod error;
pub mod prelude;

pub use api::EvaluationContext;
pub use api::EvaluationContextStats;
pub use api::EvaluationStats;
pub use api::IntegralTensor;
pub use api::Session;
pub use api::SessionQuery;
pub use api::SessionRequest;
pub use api::TypedEvaluationOutput;
pub use api::WorkspaceChunk;
pub use api::WorkspaceExecutionToken;
pub use api::WorkspacePlan;
pub use api::{BatchEvaluationOutput, BatchExecutionStats, BatchRequest};
pub use builder::SessionBuilder;
pub use error::{FacadeError, FacadeErrorKind};

#[cfg(feature = "unstable-source-api")]
pub use api::unstable;
