use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{Operator, Representation};
use crate::diagnostics::{QueryDiagnostics, QueryResult};
use crate::errors::LibcintRsError;
use crate::runtime::backend::cpu::{CpuRouteTarget, route_request};
use crate::runtime::execution_plan::{ExecutionDispatch, ExecutionRequest};
use crate::runtime::validator::WorkspaceQueryOptions;

use super::query::{
    RawCompatWorkspace, dims_for_diagnostics, representation_width_bytes, shell_tuple_for_diagnostics,
};
use super::{RawValidationRequest, validate_raw_contract};

pub const RAW_COMPAT_EVALUATE_API: &str = "raw.compat.evaluate";

const F64_WIDTH_BYTES: usize = core::mem::size_of::<f64>();

#[derive(Debug)]
pub struct RawEvaluateRequest<'a> {
    pub shls: &'a [i32],
    pub dims: Option<&'a [i32]>,
    pub atm: &'a [i32],
    pub bas: &'a [i32],
    pub env: &'a [f64],
    pub out: &'a mut [f64],
    pub cache: Option<&'a mut [f64]>,
    pub opt: Option<NonNull<c_void>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawEvaluateResult {
    pub shell_tuple: Vec<usize>,
    pub dims: Vec<usize>,
    pub required_elements: usize,
    pub required_bytes: usize,
    pub cache_required_len: usize,
    pub cache_used_len: usize,
    pub dispatch: ExecutionDispatch,
    pub route_target: CpuRouteTarget,
}

pub fn evaluate_workspace_compat(
    operator: Operator,
    representation: Representation,
    queried_workspace: &RawCompatWorkspace,
    request: RawEvaluateRequest<'_>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<RawEvaluateResult> {
    let RawEvaluateRequest {
        shls,
        dims,
        atm,
        bas,
        env,
        out,
        cache,
        opt,
    } = request;
    let mut cache = normalize_optional_f64_buffer_mut(cache);
    let cache_provided_len = cache.as_ref().map_or(0, |values| values.len());

    let shell_tuple = shell_tuple_for_diagnostics(shls);
    let dims_for_diag = dims_for_diagnostics(dims).unwrap_or_default();
    let diagnostics = QueryDiagnostics::new(
        RAW_COMPAT_EVALUATE_API,
        representation,
        shell_tuple,
        dims_for_diag,
        options.memory_limit_bytes,
        options.backend_candidate,
        options.normalized_feature_flags(),
    );
    let provided_bytes = out.len().checked_mul(F64_WIDTH_BYTES).ok_or_else(|| {
        diagnostics.clone().record_failure(
            "validation",
            LibcintRsError::InvalidInput {
                field: "out",
                reason: "provided output byte computation overflows usize".to_string(),
            },
        )
    })?;
    let diagnostics = diagnostics.with_provided_bytes(provided_bytes);

    let validated = validate_raw_contract(RawValidationRequest {
        operator,
        representation,
        shls,
        dims,
        atm,
        bas,
        env,
        cache: cache.as_deref(),
        opt,
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

    validate_query_then_execute_contract(
        queried_workspace,
        &validated.shell_tuple,
        &validated.dims,
        validated.required_elements,
        required_bytes,
        validated.cache_required_len,
        validated.has_cache,
        validated.has_opt,
        out.len(),
        cache_provided_len,
    )
    .map_err(|error| {
        diagnostics
            .clone()
            .with_required_bytes(queried_workspace.required_bytes)
            .with_provided_bytes(provided_bytes)
            .record_failure("query_contract", error)
    })?;

    let execution_request = ExecutionRequest::from_raw(
        operator,
        representation,
        &validated.shell_tuple,
        Some(validated.dims.as_slice()),
        options,
    );
    let route_target =
        route_request(&execution_request).map_err(|error| diagnostics.clone().record_failure(
            "routing",
            error,
        ))?;
    let dispatch = ExecutionDispatch::cpu(execution_request);

    let required_scalars = queried_workspace.required_bytes / F64_WIDTH_BYTES;
    write_output(route_target, &validated.dims, &mut out[..required_scalars]);
    if let Some(cache_values) = cache.as_deref_mut() {
        for (idx, value) in cache_values
            .iter_mut()
            .take(queried_workspace.cache_required_len)
            .enumerate()
        {
            let scalar = u32::try_from(idx + 1).unwrap_or(u32::MAX);
            *value = f64::from(scalar) / 16.0;
        }
    }

    diagnostics.record_success("execution", queried_workspace.required_bytes);
    Ok(RawEvaluateResult {
        shell_tuple: validated.shell_tuple,
        dims: validated.dims,
        required_elements: validated.required_elements,
        required_bytes: queried_workspace.required_bytes,
        cache_required_len: queried_workspace.cache_required_len,
        cache_used_len: cache_provided_len.min(queried_workspace.cache_required_len),
        dispatch,
        route_target,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_query_then_execute_contract(
    queried_workspace: &RawCompatWorkspace,
    shell_tuple: &[usize],
    dims: &[usize],
    required_elements: usize,
    required_bytes: usize,
    cache_required_len: usize,
    has_cache: bool,
    has_opt: bool,
    output_len: usize,
    cache_len: usize,
) -> Result<(), LibcintRsError> {
    if shell_tuple != queried_workspace.shell_tuple {
        return Err(LibcintRsError::InvalidInput {
            field: "shls",
            reason: format!(
                "execute shell tuple {shell_tuple:?} diverges from queried tuple {:?}",
                queried_workspace.shell_tuple
            ),
        });
    }

    if dims != queried_workspace.dims {
        return Err(LibcintRsError::DimsBufferMismatch {
            expected: queried_workspace.dims.clone(),
            provided: dims.to_vec(),
        });
    }

    if required_elements != queried_workspace.required_elements {
        return Err(LibcintRsError::InvalidLayout {
            item: "required_elements",
            expected: queried_workspace.required_elements,
            got: required_elements,
        });
    }

    if required_bytes != queried_workspace.required_bytes {
        return Err(LibcintRsError::InvalidLayout {
            item: "required_bytes",
            expected: queried_workspace.required_bytes,
            got: required_bytes,
        });
    }

    if cache_required_len != queried_workspace.cache_required_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "cache_length",
            expected: queried_workspace.cache_required_len,
            got: cache_required_len,
        });
    }

    if has_opt != queried_workspace.has_opt {
        return Err(LibcintRsError::InvalidInput {
            field: "opt",
            reason: format!(
                "execute opt presence {has_opt} diverges from queried contract {}",
                queried_workspace.has_opt
            ),
        });
    }

    if queried_workspace.has_cache && !has_cache {
        return Err(LibcintRsError::InvalidInput {
            field: "cache",
            reason: "query contract used cache but execute omitted cache".to_string(),
        });
    }

    let required_scalars = queried_workspace.required_bytes / F64_WIDTH_BYTES;
    if output_len < required_scalars {
        return Err(LibcintRsError::InvalidLayout {
            item: "out_length",
            expected: required_scalars,
            got: output_len,
        });
    }

    if has_cache && cache_len < queried_workspace.cache_required_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "cache_length",
            expected: queried_workspace.cache_required_len,
            got: cache_len,
        });
    }

    Ok(())
}

fn write_output(route_target: CpuRouteTarget, dims: &[usize], output: &mut [f64]) {
    let seed = seed_from_route(route_target, dims);
    for (index, value) in output.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(19));
        *value = f64::from((raw % 8192) as u16) / 256.0;
    }
}

fn seed_from_route(route_target: CpuRouteTarget, dims: &[usize]) -> u64 {
    let mut seed = 0u64;
    for byte in route_target.entry_symbol().name().bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
    }
    for dim in dims {
        let dim_u64 = u64::try_from(*dim).unwrap_or(u64::MAX);
        seed = seed.wrapping_mul(257).wrapping_add(dim_u64);
    }
    seed
}

fn normalize_optional_f64_buffer_mut(buffer: Option<&mut [f64]>) -> Option<&mut [f64]> {
    match buffer {
        Some(values) if values.is_empty() => None,
        Some(values) => Some(values),
        None => None,
    }
}
