use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::{QueryDiagnostics, QueryResult};
use crate::errors::LibcintRsError;
use crate::runtime::raw::{RawValidationRequest, validate_raw_contract};
use crate::runtime::{WorkspaceQuery, WorkspaceQueryOptions, query_workspace_raw};

const RAW_COMPAT_QUERY_API: &str = "raw.compat.query_workspace";

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawCompatWorkspace {
    pub shell_tuple: Vec<usize>,
    pub natural_dims: Vec<usize>,
    pub dims: Vec<usize>,
    pub required_elements: usize,
    pub required_bytes: usize,
    pub natm: usize,
    pub nbas: usize,
    pub env_len: usize,
    pub cache_required_len: usize,
    pub has_cache: bool,
    pub has_opt: bool,
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
    let shell_tuple = shell_tuple_for_diagnostics(request.shls);
    let dims = dims_for_diagnostics(request.dims).unwrap_or_default();
    let diagnostics = QueryDiagnostics::new(
        RAW_COMPAT_QUERY_API,
        representation,
        shell_tuple,
        dims,
        options.memory_limit_bytes,
        options.backend_candidate,
        options.normalized_feature_flags(),
    );

    let validated = validate_raw_contract(RawValidationRequest {
        operator,
        representation,
        shls: request.shls,
        dims: request.dims,
        atm: request.atm,
        bas: request.bas,
        env: request.env,
        cache: request.cache,
        opt: request.opt,
    })
    .map_err(|error| diagnostics.clone().record_failure("validation", error))?;

    let required_bytes = validated
        .required_elements
        .checked_mul(representation_width_bytes(representation))
        .ok_or_else(|| {
            diagnostics.clone().record_failure(
                "validation",
                LibcintRsError::InvalidInput {
                    field: "workspace",
                    reason: "required byte computation overflows usize".to_string(),
                },
            )
        })?;
    let diagnostics = diagnostics
        .with_dims(validated.dims.clone())
        .with_required_bytes(required_bytes);
    diagnostics.record_success("validation", required_bytes);

    Ok(RawCompatWorkspace {
        shell_tuple: validated.shell_tuple,
        natural_dims: validated.natural_dims,
        dims: validated.dims,
        required_elements: validated.required_elements,
        required_bytes,
        natm: validated.natm,
        nbas: validated.nbas,
        env_len: validated.env_len,
        cache_required_len: validated.cache_required_len,
        has_cache: validated.has_cache,
        has_opt: validated.has_opt,
    })
}

fn shell_tuple_for_diagnostics(shls: &[i32]) -> Vec<usize> {
    shls.iter()
        .filter_map(|shell_index| usize::try_from(*shell_index).ok())
        .collect()
}

fn dims_for_diagnostics(dims: Option<&[i32]>) -> Option<Vec<usize>> {
    dims.map(|provided| {
        provided
            .iter()
            .filter_map(|dim| usize::try_from(*dim).ok())
            .collect()
    })
}

fn representation_width_bytes(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    }
}
