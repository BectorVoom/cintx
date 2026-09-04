//! Stable prelude re-exports for the safe facade.
//!
//! This contract keeps stable builder/session/query/evaluate types in one import while
//! preserving the unstable source-only namespace behind explicit feature gating.

// Builder-first entrypoint for typed safe session creation.
pub use crate::builder::SessionBuilder;

// Typed request/query/evaluate entry points.
pub use crate::api::Session;
pub use crate::api::SessionQuery;
pub use crate::api::SessionRequest;
pub use crate::api::TypedEvaluationOutput;
pub use crate::api::{
    BatchEvaluationOutput, BatchExecutionStats, BatchRequest, EvaluationContext,
    EvaluationContextStats, PairBatchRequest, QuartetBatchChunk, QuartetBatchOutput,
    QuartetBatchRequest, ShellListBatchOutput, TripleBatchRequest, evaluate_shell_pairs,
    evaluate_shell_pairs_in, evaluate_shell_quartets, evaluate_shell_quartets_in,
    evaluate_shell_triples, evaluate_shell_triples_in,
};

// Structured workspace metadata and execution-token contracts.
pub use crate::api::WorkspaceChunk;
pub use crate::api::WorkspaceExecutionToken;
pub use crate::api::WorkspacePlan;

// Owned output tensor and execution statistics.
pub use crate::api::EvaluationStats;
pub use crate::api::IntegralTensor;

// Explicit stable fallback for unstable source requests.
pub use crate::api::unsupported_unstable_request;

// Stable facade error taxonomy.
pub use crate::error::FacadeError;
pub use crate::error::FacadeErrorKind;

// Core typed inputs accepted by SessionBuilder and SessionRequest.
pub use cintx_core::AoSymmetry;
pub use cintx_core::BasisSet;
pub use cintx_core::OperatorId;
pub use cintx_core::Representation;
pub use cintx_core::ShellTuple;

// Runtime options consumed by the safe session builder/request contract.
pub use cintx_runtime::ExecutionOptions;

#[cfg(feature = "unstable-source-api")]
pub use crate::api::unstable;
