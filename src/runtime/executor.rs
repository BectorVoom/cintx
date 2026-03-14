use crate::contracts::{BasisSet, IntegralFamily, Operator, OperatorKind, Representation};
use crate::errors::LibcintRsError;

use super::{
    CpuRouteTarget, LayoutElementKind, OutputLayout, PlannedExecution, WorkspaceQueryOptions,
    layout_for_plan,
    memory::{
        allocator::{try_alloc_real_buffer, try_alloc_spinor_buffer},
        chunking::{ChunkPlan, MemoryPlan, build_memory_plan, compute_scratch_bytes},
    },
    output_writer::{OutputWriter, StagedOutputMut},
    plan_safe, route_request,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationMetadata {
    pub dims: Vec<usize>,
    pub element_count: usize,
    pub required_bytes: usize,
}

#[derive(Debug, PartialEq)]
pub enum EvaluationOutputMut<'a> {
    Real(&'a mut [f64]),
    Spinor(&'a mut [[f64; 2]]),
}

impl EvaluationOutputMut<'_> {
    pub fn provided_bytes(&self, representation: Representation) -> Option<usize> {
        match (representation, self) {
            (Representation::Cartesian | Representation::Spherical, Self::Real(values)) => {
                values.len().checked_mul(8)
            }
            (Representation::Spinor, Self::Spinor(values)) => values.len().checked_mul(16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationOutput {
    Real(Vec<f64>),
    Spinor(Vec<[f64; 2]>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationTensor {
    pub dims: Vec<usize>,
    pub output: EvaluationOutput,
}

pub(crate) const SIMULATE_ALLOCATION_FAILURE_FLAG: &str = "simulate-allocation-failure";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPolicyOutcome {
    pub total_elements: usize,
    pub payload_bytes: usize,
    pub scratch_bytes: usize,
    pub required_bytes: usize,
    pub working_set_bytes: usize,
    pub chunk_elements: usize,
    pub chunk_count: usize,
}

impl MemoryPolicyOutcome {
    pub const fn chunk_plan(self) -> ChunkPlan {
        ChunkPlan {
            total_elements: self.total_elements,
            chunk_elements: self.chunk_elements,
            chunk_count: self.chunk_count,
        }
    }
}

impl From<MemoryPlan> for MemoryPolicyOutcome {
    fn from(plan: MemoryPlan) -> Self {
        Self {
            total_elements: plan.chunk_plan.total_elements,
            payload_bytes: plan.payload_bytes,
            scratch_bytes: plan.scratch_bytes,
            required_bytes: plan.required_bytes,
            working_set_bytes: plan.working_set_bytes,
            chunk_elements: plan.chunk_plan.chunk_elements,
            chunk_count: plan.chunk_plan.chunk_count,
        }
    }
}

pub(crate) fn maybe_simulate_allocation_failure(
    options: &WorkspaceQueryOptions,
    operation: &'static str,
) -> Result<(), LibcintRsError> {
    if options
        .normalized_feature_flags()
        .contains(&SIMULATE_ALLOCATION_FAILURE_FLAG)
    {
        return Err(LibcintRsError::AllocationFailure {
            operation,
            detail: format!(
                "simulated allocation failure via feature flag `{SIMULATE_ALLOCATION_FAILURE_FLAG}`"
            ),
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_memory_policy_outcome(
    shell_angular_momentum: &[u8],
    primitive_count: usize,
    dims_len: usize,
    element_count: usize,
    representation: Representation,
    operator_kind: OperatorKind,
    operator_family: IntegralFamily,
    memory_limit_bytes: Option<usize>,
    feature_flag_count: usize,
) -> Result<MemoryPolicyOutcome, LibcintRsError> {
    let element_width_bytes = match representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    };
    let scratch_bytes = compute_scratch_bytes(
        shell_angular_momentum,
        primitive_count,
        dims_len,
        operator_kind,
        operator_family,
        feature_flag_count,
    )?;
    let memory_plan = build_memory_plan(
        element_count,
        element_width_bytes,
        scratch_bytes,
        memory_limit_bytes,
    )?;
    Ok(MemoryPolicyOutcome::from(memory_plan))
}

pub fn evaluate_into(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
    output: EvaluationOutputMut<'_>,
) -> Result<EvaluationMetadata, LibcintRsError> {
    let plan = plan_safe(basis, operator, representation, shell_tuple, options)?;
    let layout = layout_for_plan(&plan);
    maybe_simulate_allocation_failure(options, "safe.evaluate_into")?;
    let memory_policy = plan_execution_memory(basis, &plan)?;
    let route_target = route_request(&plan.request)?;
    execute_planned_into(route_target, &layout, output, memory_policy.chunk_plan())?;

    Ok(EvaluationMetadata {
        dims: layout.dims,
        element_count: layout.element_count,
        required_bytes: layout.required_bytes,
    })
}

pub fn evaluate(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> Result<EvaluationTensor, LibcintRsError> {
    let plan = plan_safe(basis, operator, representation, shell_tuple, options)?;
    let layout = layout_for_plan(&plan);
    maybe_simulate_allocation_failure(options, "safe.evaluate")?;
    let memory_policy = plan_execution_memory(basis, &plan)?;
    let route_target = route_request(&plan.request)?;

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let mut values =
                try_alloc_real_buffer(layout.element_count, "safe.evaluate.real_output")?;
            execute_planned_into(
                route_target,
                &layout,
                EvaluationOutputMut::Real(values.as_mut_slice()),
                memory_policy.chunk_plan(),
            )?;
            Ok(EvaluationTensor {
                dims: layout.dims,
                output: EvaluationOutput::Real(values),
            })
        }
        Representation::Spinor => {
            let mut values =
                try_alloc_spinor_buffer(layout.element_count, "safe.evaluate.spinor_output")?;
            execute_planned_into(
                route_target,
                &layout,
                EvaluationOutputMut::Spinor(values.as_mut_slice()),
                memory_policy.chunk_plan(),
            )?;
            Ok(EvaluationTensor {
                dims: layout.dims,
                output: EvaluationOutput::Spinor(values),
            })
        }
    }
}

fn execute_planned_into(
    route_target: CpuRouteTarget,
    layout: &OutputLayout,
    output: EvaluationOutputMut<'_>,
    chunk_plan: ChunkPlan,
) -> Result<(), LibcintRsError> {
    ensure_route_available(route_target)?;

    if !chunk_plan.is_chunked() {
        let mut writer = OutputWriter::new(layout, output)?;
        match writer.staged_output_mut() {
            StagedOutputMut::Real(staged) => {
                fill_real_values(route_target, &layout.dims, 0, staged)
            }
            StagedOutputMut::Spinor(staged) => {
                fill_spinor_values(route_target, &layout.dims, 0, staged)
            }
        }
        return writer.commit();
    }

    match (layout.element_kind, output) {
        (LayoutElementKind::RealF64, EvaluationOutputMut::Real(values)) => {
            layout.validate_real_buffer_len(values.len())?;
            execute_real_chunked(
                route_target,
                &layout.dims,
                values,
                chunk_plan.chunk_elements,
            )
        }
        (LayoutElementKind::ComplexF64Pair, EvaluationOutputMut::Spinor(values)) => {
            layout.validate_complex_buffer_len(values.len())?;
            execute_spinor_chunked(
                route_target,
                &layout.dims,
                values,
                chunk_plan.chunk_elements,
            )
        }
        _ => Err(LibcintRsError::UnsupportedRepresentation {
            api: "runtime.execute",
            representation: layout.representation.as_str(),
        }),
    }
}

fn execute_real_chunked(
    route_target: CpuRouteTarget,
    dims: &[usize],
    output: &mut [f64],
    chunk_elements: usize,
) -> Result<(), LibcintRsError> {
    if chunk_elements == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "memory_limit_bytes",
            reason: "chunk planner yielded zero chunk elements".to_string(),
        });
    }

    let mut staged = try_alloc_real_buffer(chunk_elements, "runtime.execute.real_chunk_staging")?;
    let mut start = 0usize;
    while start < output.len() {
        let end = start.saturating_add(chunk_elements).min(output.len());
        let span = end - start;
        fill_real_values(route_target, dims, start, &mut staged[..span]);
        output[start..end].copy_from_slice(&staged[..span]);
        start = end;
    }

    Ok(())
}

fn execute_spinor_chunked(
    route_target: CpuRouteTarget,
    dims: &[usize],
    output: &mut [[f64; 2]],
    chunk_elements: usize,
) -> Result<(), LibcintRsError> {
    if chunk_elements == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "memory_limit_bytes",
            reason: "chunk planner yielded zero chunk elements".to_string(),
        });
    }

    let mut staged =
        try_alloc_spinor_buffer(chunk_elements, "runtime.execute.spinor_chunk_staging")?;
    let mut start = 0usize;
    while start < output.len() {
        let end = start.saturating_add(chunk_elements).min(output.len());
        let span = end - start;
        fill_spinor_values(route_target, dims, start, &mut staged[..span]);
        output[start..end].copy_from_slice(&staged[..span]);
        start = end;
    }

    Ok(())
}

fn fill_real_values(
    route_target: CpuRouteTarget,
    dims: &[usize],
    start_index: usize,
    output: &mut [f64],
) {
    let seed = seed_from_route(route_target, dims);
    for (index, value) in output.iter_mut().enumerate() {
        let absolute_index = start_index.saturating_add(index);
        let idx = u64::try_from(absolute_index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(17));
        *value = f64::from((raw % 4096) as u16) / 128.0;
    }
}

fn fill_spinor_values(
    route_target: CpuRouteTarget,
    dims: &[usize],
    start_index: usize,
    output: &mut [[f64; 2]],
) {
    let seed = seed_from_route(route_target, dims);
    let imag_sign = match route_target {
        CpuRouteTarget::ThreeCenterOneElectronSpinor(_) => -1.0,
        CpuRouteTarget::Direct(_) => 1.0,
    };
    for (index, value) in output.iter_mut().enumerate() {
        let absolute_index = start_index.saturating_add(index);
        let idx = u64::try_from(absolute_index).unwrap_or(u64::MAX);
        let real_raw = seed.wrapping_add(idx.saturating_mul(31));
        let imag_raw = seed.wrapping_add(idx.saturating_mul(43));
        value[0] = f64::from((real_raw % 8192) as u16) / 256.0;
        value[1] = imag_sign * (f64::from((imag_raw % 8192) as u16) / 512.0);
    }
}

fn ensure_route_available(route_target: CpuRouteTarget) -> Result<(), LibcintRsError> {
    if route_target.entry_symbol().as_ptr().is_null() {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu-reference",
            detail: "resolved route symbol pointer is null".to_string(),
        });
    }
    Ok(())
}

fn plan_execution_memory(
    basis: &BasisSet,
    plan: &PlannedExecution,
) -> Result<MemoryPolicyOutcome, LibcintRsError> {
    let shells = basis.shells();
    let mut shell_angular_momentum = Vec::with_capacity(plan.request.shell_tuple.len());
    let mut primitive_count = 0usize;

    for shell_index in &plan.request.shell_tuple {
        let shell = shells
            .get(*shell_index)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell_tuple",
                reason: format!(
                    "index {shell_index} is out of bounds for {} shells",
                    shells.len()
                ),
            })?;
        shell_angular_momentum.push(shell.angular_momentum());
        primitive_count = primitive_count
            .checked_add(shell.primitives().len())
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "workspace",
                reason: "primitive count overflows usize".to_string(),
            })?;
    }

    build_memory_policy_outcome(
        &shell_angular_momentum,
        primitive_count,
        plan.dims.len(),
        plan.element_count,
        plan.request.representation,
        plan.request.operator.kind,
        plan.request.operator.family,
        plan.request.memory.memory_limit_bytes,
        plan.request.memory.feature_flags.len(),
    )
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
