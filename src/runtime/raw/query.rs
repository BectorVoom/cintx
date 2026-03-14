use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{Operator, Representation};
use crate::diagnostics::{QueryDiagnostics, QueryResult};
use crate::errors::LibcintRsError;
use crate::runtime::executor::build_memory_policy_outcome;
use crate::runtime::validator::WorkspaceQueryOptions;

use super::{RawValidationRequest, validate_raw_contract};

pub const RAW_COMPAT_QUERY_API: &str = "raw.compat.query_workspace";

const F64_WIDTH_BYTES: usize = core::mem::size_of::<f64>();

#[derive(Debug, Clone, Copy)]
pub struct RawQueryRequest<'a> {
    pub shls: &'a [i32],
    pub dims: Option<&'a [i32]>,
    pub atm: &'a [i32],
    pub bas: &'a [i32],
    pub env: &'a [f64],
    pub out: Option<&'a [f64]>,
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
    pub memory_required_bytes: usize,
    pub memory_working_set_bytes: usize,
    pub memory_scratch_bytes: usize,
    pub chunk_elements: usize,
    pub chunk_count: usize,
    pub natm: usize,
    pub nbas: usize,
    pub env_len: usize,
    pub cache_required_len: usize,
    pub has_cache: bool,
    pub has_opt: bool,
    pub query_uses_null_out: bool,
    pub query_uses_null_cache: bool,
    pub output_provided_len: usize,
    pub cache_provided_len: usize,
}

pub fn query_workspace_compat(
    operator: Operator,
    representation: Representation,
    request: RawQueryRequest<'_>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<RawCompatWorkspace> {
    let request = RawQueryRequest {
        out: normalize_optional_f64_buffer(request.out),
        cache: normalize_optional_f64_buffer(request.cache),
        ..request
    };
    let shell_tuple = shell_tuple_for_diagnostics(request.shls);
    let dims = dims_for_diagnostics(request.dims).unwrap_or_default();
    let output_provided_len = request.out.map_or(0, <[f64]>::len);
    let cache_provided_len = request.cache.map_or(0, <[f64]>::len);
    let query_uses_null_out = request.out.is_none();
    let query_uses_null_cache = request.cache.is_none();

    let diagnostics = QueryDiagnostics::new(
        RAW_COMPAT_QUERY_API,
        representation,
        shell_tuple,
        dims,
        options.memory_limit_bytes,
        options.backend_candidate,
        options.normalized_feature_flags(),
    );
    let output_provided_bytes =
        output_provided_len
            .checked_mul(F64_WIDTH_BYTES)
            .ok_or_else(|| {
                diagnostics.clone().record_failure(
                    "validation",
                    LibcintRsError::InvalidInput {
                        field: "out",
                        reason: "provided output byte computation overflows usize".to_string(),
                    },
                )
            })?;
    let diagnostics = if query_uses_null_out {
        diagnostics
    } else {
        diagnostics.with_provided_bytes(output_provided_bytes)
    };

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
    let memory_policy = build_memory_policy_outcome(
        &validated.shell_angular_momentum,
        validated.primitive_count,
        validated.dims.len(),
        validated.required_elements,
        representation,
        operator.kind(),
        operator.family(),
        options.memory_limit_bytes,
        options.normalized_feature_flags().len(),
    )
    .map_err(|error| diagnostics.clone().record_failure("memory_policy", error))?;

    let diagnostics = diagnostics
        .with_dims(validated.dims.clone())
        .with_required_bytes(memory_policy.required_bytes);
    diagnostics.record_success("validation", memory_policy.required_bytes);

    Ok(RawCompatWorkspace {
        shell_tuple: validated.shell_tuple,
        natural_dims: validated.natural_dims,
        dims: validated.dims,
        required_elements: validated.required_elements,
        required_bytes,
        memory_required_bytes: memory_policy.required_bytes,
        memory_working_set_bytes: memory_policy.working_set_bytes,
        memory_scratch_bytes: memory_policy.scratch_bytes,
        chunk_elements: memory_policy.chunk_elements,
        chunk_count: memory_policy.chunk_count,
        natm: validated.natm,
        nbas: validated.nbas,
        env_len: validated.env_len,
        cache_required_len: validated.cache_required_len,
        has_cache: validated.has_cache,
        has_opt: validated.has_opt,
        query_uses_null_out,
        query_uses_null_cache,
        output_provided_len,
        cache_provided_len,
    })
}

pub(crate) fn shell_tuple_for_diagnostics(shls: &[i32]) -> Vec<usize> {
    shls.iter()
        .filter_map(|shell_index| usize::try_from(*shell_index).ok())
        .collect()
}

pub(crate) fn dims_for_diagnostics(dims: Option<&[i32]>) -> Option<Vec<usize>> {
    dims.map(|provided| {
        provided
            .iter()
            .filter_map(|dim| usize::try_from(*dim).ok())
            .collect()
    })
}

pub(crate) fn representation_width_bytes(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    }
}

fn normalize_optional_f64_buffer(buffer: Option<&[f64]>) -> Option<&[f64]> {
    match buffer {
        Some(values) if values.is_empty() => None,
        Some(values) => Some(values),
        None => None,
    }
}
