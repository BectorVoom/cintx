#![deny(unsafe_op_in_unsafe_fn)]

pub mod contracts;
pub mod diagnostics;
pub mod errors;
pub mod runtime;

pub use contracts::{
    Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, Shell, ShellPrimitive,
    validate_dims,
};
pub use diagnostics::{QueryDiagnostics, QueryError, QueryResult};
pub use errors::LibcintRsError;
pub use runtime::{
    ValidatedInputs, ValidatedShape, WorkspaceQuery, WorkspaceQueryOptions, query_workspace_raw,
    query_workspace_safe,
};
