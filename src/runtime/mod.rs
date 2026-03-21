pub mod backend;
pub mod execution_plan;
pub mod executor;
pub mod helpers;
pub mod layout;
pub mod memory;
pub mod output_writer;
pub mod planner;
mod policy;
pub mod raw;
pub mod validator;
pub mod workspace_query;

pub use backend::cpu::{
    ALL_BOUND_SYMBOLS, CpuKernelFn, CpuKernelSymbol, CpuRouteKey, CpuRouteManifestEntry,
    CpuRouteTarget, ResolvedCpuRoute, RouteEntryKernel, RouteKind, RouteOptimizerMode,
    RouteStability, RouteStatus, RouteSurface, RouteSurfaceGroup, Spinor3c1eAdapter,
    Spinor3c1eTransform, adapter_route, resolve_capi_route, resolve_raw_route, resolve_route,
    resolve_route_request, resolve_safe_route, route, route_manifest_entries,
    route_manifest_lock_json, route_request,
};
pub use execution_plan::{
    ExecutionBackend, ExecutionDispatch, ExecutionMemoryOptions, ExecutionOperator,
    ExecutionRequest,
};
pub use executor::{
    EvaluationMetadata, EvaluationOutput, EvaluationOutputMut, EvaluationTensor, evaluate,
    evaluate_into,
};
pub use helpers::{
    ShellAoLayout, ShellNormalizationMetadata, cartesian_component_count, contracted_shell_count,
    deterministic_transform_scalars, gto_norm, shell_ao_counts, shell_ao_layout,
    shell_normalization_metadata, shell_offsets, spherical_component_count, spinor_component_count,
    total_ao_count,
};
pub use layout::{LayoutElementKind, OutputLayout, layout_for_plan};
pub use planner::{PlannedExecution, plan_execution, plan_raw, plan_safe};
pub use raw::{
    RAW_COMPAT_EVALUATE_API, RAW_COMPAT_QUERY_API, RawCompatWorkspace, RawEvaluateRequest,
    RawEvaluateResult, RawQueryRequest, evaluate_workspace_compat, query_workspace_compat,
};
pub use validator::{
    ValidatedInputs, ValidatedShape, WorkspaceQueryOptions, validate_raw_query_inputs,
    validate_safe_query_inputs,
};
pub use workspace_query::{WorkspaceQuery, query_workspace_raw, query_workspace_safe};
