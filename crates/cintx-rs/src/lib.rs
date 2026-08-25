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
pub use api::{
    BatchEvaluationOutput, BatchExecutionStats, BatchRequest, PairBatchRequest, QuartetBatchOutput,
    QuartetBatchRequest, ShellListBatchOutput, TripleBatchRequest, evaluate_shell_pairs,
    evaluate_shell_pairs_in, evaluate_shell_quartets, evaluate_shell_quartets_in,
    evaluate_shell_triples, evaluate_shell_triples_in,
};
pub use builder::SessionBuilder;
pub use error::{FacadeError, FacadeErrorKind};

#[cfg(feature = "unstable-source-api")]
pub use api::unstable;
