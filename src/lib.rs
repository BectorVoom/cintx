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
    CanonicalSymbolIdentity, CompiledManifestLock, FamilyTag, LockUpdateApproval, LockUpdateReason,
    ManifestGovernanceError, ManifestLockEntry, ManifestProfile, OperatorTag, RepresentationTag,
    StabilityClass,
};
pub use runtime::{
    ALL_BOUND_SYMBOLS, CpuKernelFn, CpuKernelSymbol, CpuRouteKey, CpuRouteTarget,
    EvaluationMetadata, EvaluationOutput, EvaluationOutputMut, EvaluationTensor, ExecutionBackend,
    ExecutionDispatch, ExecutionMemoryOptions, ExecutionOperator, ExecutionRequest,
    LayoutElementKind, OutputLayout, PlannedExecution, RAW_COMPAT_EVALUATE_API,
    RAW_COMPAT_QUERY_API, RawCompatWorkspace, RawEvaluateRequest, RawEvaluateResult,
    RawQueryRequest, ShellAoLayout, ShellNormalizationMetadata, Spinor3c1eAdapter,
    Spinor3c1eTransform, ValidatedInputs, ValidatedShape, WorkspaceQuery, WorkspaceQueryOptions,
    adapter_route, cartesian_component_count, contracted_shell_count,
    deterministic_transform_scalars, evaluate, evaluate_into, evaluate_workspace_compat, gto_norm,
    layout_for_plan, plan_execution, plan_raw, plan_safe, query_workspace_compat,
    query_workspace_raw, query_workspace_safe, route, route_request, shell_ao_counts,
    shell_ao_layout, shell_normalization_metadata, shell_offsets, spherical_component_count,
    spinor_component_count, total_ao_count,
};
