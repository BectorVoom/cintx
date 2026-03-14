use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::QueryResult;
use crate::runtime::raw::{
    evaluate_workspace_compat as runtime_evaluate_workspace_compat,
    query_workspace_compat as runtime_query_workspace_compat,
};
use crate::runtime::{WorkspaceQuery, WorkspaceQueryOptions, query_workspace_raw};

pub use crate::runtime::raw::{
    RawCompatWorkspace, RawEvaluateRequest, RawEvaluateResult, RawQueryRequest,
};

#[derive(Debug, Clone, Copy)]
pub struct RawCompatRequest<'a> {
    pub shls: &'a [i32],
    pub dims: Option<&'a [i32]>,
    pub atm: &'a [i32],
    pub bas: &'a [i32],
    pub env: &'a [f64],
    pub cache: Option<&'a [f64]>,
    pub opt: Option<NonNull<c_void>>,
}

pub fn query_workspace(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims_override: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    query_workspace_raw(
        basis,
        operator,
        representation,
        shell_tuple,
        dims_override,
        options,
    )
}

pub fn query_workspace_compat(
    operator: Operator,
    representation: Representation,
    request: RawCompatRequest<'_>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<RawCompatWorkspace> {
    runtime_query_workspace_compat(
        operator,
        representation,
        RawQueryRequest {
            shls: request.shls,
            dims: request.dims,
            atm: request.atm,
            bas: request.bas,
            env: request.env,
            out: None,
            cache: request.cache,
            opt: request.opt,
        },
        options,
    )
}

pub fn query_workspace_compat_with_sentinels(
    operator: Operator,
    representation: Representation,
    request: RawQueryRequest<'_>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<RawCompatWorkspace> {
    runtime_query_workspace_compat(operator, representation, request, options)
}

pub fn evaluate_compat(
    operator: Operator,
    representation: Representation,
    queried_workspace: &RawCompatWorkspace,
    request: RawEvaluateRequest<'_>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<RawEvaluateResult> {
    runtime_evaluate_workspace_compat(operator, representation, queried_workspace, request, options)
}
