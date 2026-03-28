//! Safe Rust facade scaffolding for query/evaluate session flows.

use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple, cintxRsError};
use cintx_runtime::{ExecutionOptions, WorkspaceQuery, query_workspace};
use thiserror::Error;

/// Stable facade-level error mapping for the safe surface.
#[derive(Debug, Error)]
pub enum FacadeError {
    #[error(transparent)]
    Core(#[from] cintxRsError),
    #[error("unsupported api requested={requested}")]
    UnsupportedApi { requested: String },
}

/// Typed safe request object that keeps `query_workspace()` and `evaluate()` connected.
#[derive(Clone, Debug)]
pub struct SessionRequest<'basis> {
    pub operator: OperatorId,
    pub representation: Representation,
    pub basis: &'basis BasisSet,
    pub shells: ShellTuple,
    pub options: ExecutionOptions,
}

impl<'basis> SessionRequest<'basis> {
    pub fn query_workspace(&self) -> Result<SessionQuery, FacadeError> {
        let workspace = query_workspace(
            self.operator,
            self.representation,
            self.basis,
            self.shells.clone(),
            &self.options,
        )?;

        Ok(SessionQuery {
            operator: self.operator,
            representation: self.representation,
            shells: self.shells.clone(),
            options: self.options.clone(),
            workspace,
        })
    }
}

/// Result of `query_workspace()` that carries the validated request metadata forward to evaluate.
#[derive(Clone, Debug)]
pub struct SessionQuery {
    pub operator: OperatorId,
    pub representation: Representation,
    pub shells: ShellTuple,
    pub options: ExecutionOptions,
    pub workspace: WorkspaceQuery,
}

/// Placeholder for future typed outputs once evaluation wiring lands in later plans.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedEvaluationOutput {
    pub values: Vec<f64>,
}

impl SessionQuery {
    pub fn evaluate(self) -> Result<TypedEvaluationOutput, FacadeError> {
        Err(FacadeError::UnsupportedApi {
            requested: "safe evaluate() behavior is scaffolded; implementation follows in later Phase 3 plans"
                .to_owned(),
        })
    }
}

/// Explicit fallback used when unstable source requests are attempted without feature support.
pub fn unsupported_unstable_request(symbol: &str) -> FacadeError {
    FacadeError::UnsupportedApi {
        requested: format!(
            "unstable source symbol `{symbol}` requires feature `unstable-source-api`"
        ),
    }
}

#[cfg(feature = "unstable-source-api")]
pub mod unstable {
    /// Marker payload for source-only API entries that are not part of the stable facade namespace.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SourceApiToken {
        pub family: &'static str,
        pub symbol: &'static str,
    }

    impl SourceApiToken {
        pub const fn new(family: &'static str, symbol: &'static str) -> Self {
            Self { family, symbol }
        }
    }
}
