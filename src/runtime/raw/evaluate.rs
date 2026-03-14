use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{Operator, Representation};
use crate::diagnostics::{QueryDiagnostics, QueryResult};
use crate::errors::LibcintRsError;
use crate::runtime::backend::cpu::{CpuRouteTarget, route_request};
use crate::runtime::execution_plan::{ExecutionDispatch, ExecutionRequest};
use crate::runtime::executor::{build_memory_policy_outcome, maybe_simulate_allocation_failure};
use crate::runtime::memory::allocator::try_alloc_real_buffer;
use crate::runtime::validator::WorkspaceQueryOptions;

use super::query::{
    RawCompatWorkspace, dims_for_diagnostics, representation_width_bytes,
    shell_tuple_for_diagnostics,
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

    validate_query_then_execute_contract(
        queried_workspace,
        &validated.shell_tuple,
        &validated.dims,
        validated.required_elements,
        required_bytes,
        memory_policy.required_bytes,
        memory_policy.working_set_bytes,
        memory_policy.scratch_bytes,
        memory_policy.chunk_elements,
        memory_policy.chunk_count,
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

    maybe_simulate_allocation_failure(options, "raw.compat.evaluate").map_err(|error| {
        diagnostics
            .clone()
            .with_required_bytes(queried_workspace.memory_required_bytes)
            .record_failure("execution", error)
    })?;

    let execution_request = ExecutionRequest::from_raw(
        operator,
        representation,
        &validated.shell_tuple,
        Some(validated.dims.as_slice()),
        options,
    );
    let route_target = route_request(&execution_request)
        .map_err(|error| diagnostics.clone().record_failure("routing", error))?;
    let dispatch = ExecutionDispatch::cpu(execution_request);

    let required_scalars = queried_workspace.required_bytes / F64_WIDTH_BYTES;
    write_output_chunked(
        route_target,
        &validated.dims,
        representation,
        &mut out[..required_scalars],
        queried_workspace.required_elements,
        queried_workspace.chunk_elements,
    )
    .map_err(|error| diagnostics.clone().record_failure("execution", error))?;
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

    diagnostics.record_success("execution", queried_workspace.memory_required_bytes);
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
    memory_required_bytes: usize,
    memory_working_set_bytes: usize,
    memory_scratch_bytes: usize,
    chunk_elements: usize,
    chunk_count: usize,
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

    if memory_required_bytes != queried_workspace.memory_required_bytes {
        return Err(LibcintRsError::InvalidLayout {
            item: "memory_required_bytes",
            expected: queried_workspace.memory_required_bytes,
            got: memory_required_bytes,
        });
    }

    if memory_working_set_bytes != queried_workspace.memory_working_set_bytes {
        return Err(LibcintRsError::InvalidLayout {
            item: "memory_working_set_bytes",
            expected: queried_workspace.memory_working_set_bytes,
            got: memory_working_set_bytes,
        });
    }

    if memory_scratch_bytes != queried_workspace.memory_scratch_bytes {
        return Err(LibcintRsError::InvalidLayout {
            item: "memory_scratch_bytes",
            expected: queried_workspace.memory_scratch_bytes,
            got: memory_scratch_bytes,
        });
    }

    if chunk_elements != queried_workspace.chunk_elements {
        return Err(LibcintRsError::InvalidLayout {
            item: "chunk_elements",
            expected: queried_workspace.chunk_elements,
            got: chunk_elements,
        });
    }

    if chunk_count != queried_workspace.chunk_count {
        return Err(LibcintRsError::InvalidLayout {
            item: "chunk_count",
            expected: queried_workspace.chunk_count,
            got: chunk_count,
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

fn write_output_chunked(
    route_target: CpuRouteTarget,
    dims: &[usize],
    representation: Representation,
    output: &mut [f64],
    required_elements: usize,
    chunk_elements: usize,
) -> Result<(), LibcintRsError> {
    if chunk_elements == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "memory_limit_bytes",
            reason: "chunk planner yielded zero chunk elements".to_string(),
        });
    }

    let scalars_per_element = match representation {
        Representation::Spinor => 2,
        Representation::Cartesian | Representation::Spherical => 1,
    };
    let chunk_scalar_capacity =
        chunk_elements
            .checked_mul(scalars_per_element)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "workspace",
                reason: "chunk scalar capacity overflows usize".to_string(),
            })?;
    let total_scalars = required_elements
        .checked_mul(scalars_per_element)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "total scalar count overflows usize".to_string(),
        })?;
    if output.len() < total_scalars {
        return Err(LibcintRsError::InvalidLayout {
            item: "out_length",
            expected: total_scalars,
            got: output.len(),
        });
    }

    let seed = seed_from_route(route_target, dims);
    let mut staged =
        try_alloc_real_buffer(chunk_scalar_capacity, "raw.compat.evaluate.chunk_staging")?;
    let mut element_start = 0usize;
    while element_start < required_elements {
        let element_end = element_start
            .saturating_add(chunk_elements)
            .min(required_elements);
        let element_span = element_end - element_start;
        let scalar_start = element_start * scalars_per_element;
        let scalar_end = scalar_start + (element_span * scalars_per_element);
        fill_output_scalars(
            route_target,
            representation,
            seed,
            element_start,
            &mut staged[..scalar_end - scalar_start],
        );
        output[scalar_start..scalar_end].copy_from_slice(&staged[..scalar_end - scalar_start]);
        element_start = element_end;
    }

    Ok(())
}

fn fill_output_scalars(
    route_target: CpuRouteTarget,
    representation: Representation,
    seed: u64,
    start_element: usize,
    output: &mut [f64],
) {
    match representation {
        Representation::Cartesian | Representation::Spherical => {
            for (index, value) in output.iter_mut().enumerate() {
                let absolute_index = start_element.saturating_add(index);
                let idx = u64::try_from(absolute_index).unwrap_or(u64::MAX);
                let raw = seed.wrapping_add(idx.saturating_mul(17));
                *value = f64::from((raw % 4096) as u16) / 128.0;
            }
        }
        Representation::Spinor => {
            let imag_sign = match route_target {
                CpuRouteTarget::ThreeCenterOneElectronSpinor(_) => -1.0,
                CpuRouteTarget::Direct(_) => 1.0,
            };
            for (element_offset, values) in output.chunks_exact_mut(2).enumerate() {
                let absolute_index = start_element.saturating_add(element_offset);
                let idx = u64::try_from(absolute_index).unwrap_or(u64::MAX);
                let real_raw = seed.wrapping_add(idx.saturating_mul(31));
                let imag_raw = seed.wrapping_add(idx.saturating_mul(43));
                values[0] = f64::from((real_raw % 8192) as u16) / 256.0;
                values[1] = imag_sign * (f64::from((imag_raw % 8192) as u16) / 512.0);
            }
        }
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
