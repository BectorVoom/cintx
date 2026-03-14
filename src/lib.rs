#![deny(unsafe_op_in_unsafe_fn)]

pub mod api;
pub mod contracts;
pub mod diagnostics;
pub mod errors;
pub mod runtime;

pub use api::{raw, safe};
pub use contracts::{
    Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, Shell, ShellPrimitive,
    validate_dims,
};
pub use diagnostics::{QueryDiagnostics, QueryError, QueryResult};
pub use errors::LibcintRsError;
pub use runtime::{
    ALL_BOUND_SYMBOLS, CpuKernelFn, CpuKernelSymbol, CpuRouteKey, CpuRouteTarget, ExecutionBackend,
    ExecutionDispatch, ExecutionMemoryOptions, ExecutionOperator, ExecutionRequest,
    ValidatedInputs, ValidatedShape, WorkspaceQuery, WorkspaceQueryOptions, query_workspace_raw,
    query_workspace_safe, route, route_request,
};
