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
    EvaluationMetadata, EvaluationOutput, EvaluationOutputMut, EvaluationTensor,
    LayoutElementKind, OutputLayout, PlannedExecution, Spinor3c1eAdapter, Spinor3c1eTransform,
    ValidatedInputs, ValidatedShape, WorkspaceQuery, WorkspaceQueryOptions, adapter_route,
    evaluate, evaluate_into, layout_for_plan, plan_execution, plan_raw, plan_safe,
    query_workspace_raw, query_workspace_safe, route, route_request,
};
