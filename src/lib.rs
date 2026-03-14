#![deny(unsafe_op_in_unsafe_fn)]

pub mod api;
pub mod contracts;
pub mod diagnostics;
pub mod errors;
pub mod manifest;
pub mod runtime;

pub use api::{raw, safe};
pub use contracts::{
    Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, Shell, ShellPrimitive,
    validate_dims,
};
pub use diagnostics::{QueryDiagnostics, QueryError, QueryResult};
pub use errors::LibcintRsError;
pub use manifest::{
    CanonicalSymbolIdentity, CompiledManifestLock, FamilyTag, LockUpdateApproval,
    LockUpdateReason, ManifestGovernanceError, ManifestLockEntry, ManifestProfile, OperatorTag,
    RepresentationTag, StabilityClass,
};
pub use runtime::{
    ALL_BOUND_SYMBOLS, CpuKernelFn, CpuKernelSymbol, CpuRouteKey, CpuRouteTarget, ExecutionBackend,
    ExecutionDispatch, ExecutionMemoryOptions, ExecutionOperator, ExecutionRequest,
    EvaluationMetadata, EvaluationOutput, EvaluationOutputMut, EvaluationTensor,
    LayoutElementKind, OutputLayout, PlannedExecution, Spinor3c1eAdapter, Spinor3c1eTransform,
    RAW_COMPAT_EVALUATE_API, RAW_COMPAT_QUERY_API, RawCompatWorkspace, RawEvaluateRequest,
    RawEvaluateResult, RawQueryRequest, ValidatedInputs, ValidatedShape, WorkspaceQuery,
    WorkspaceQueryOptions, adapter_route, evaluate, evaluate_into, evaluate_workspace_compat,
    layout_for_plan, plan_execution, plan_raw, plan_safe, query_workspace_compat,
    query_workspace_raw, query_workspace_safe, route, route_request,
};
