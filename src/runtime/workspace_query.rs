use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::QueryResult;
use crate::errors::LibcintRsError;

use super::memory::chunking::{build_memory_plan, compute_scratch_bytes};
use super::validator::{
    ValidatedInputs, ValidatedShape, WorkspaceQueryOptions, make_query_diagnostics,
    validate_raw_query_inputs, validate_safe_query_inputs,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceQuery {
    pub required_bytes: usize,
    pub alignment_bytes: usize,
    pub natural_dims: Vec<usize>,
    pub dims: Vec<usize>,
    pub element_count: usize,
    pub scratch_bytes: usize,
}

pub fn query_workspace_safe(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    let diagnostics = make_query_diagnostics(
        "safe.query_workspace",
        representation,
        shell_tuple,
        None,
        options,
    );
    let (validated_inputs, validated_shape) =
        validate_safe_query_inputs(basis, operator, representation, shell_tuple, options)
            .map_err(|error| diagnostics.clone().record_failure("validation", error))?;
    let diagnostics = diagnostics.with_dims(validated_shape.dims.clone());
    let workspace =
        estimate_workspace(&validated_inputs, &validated_shape, options).map_err(|error| {
            diagnostics
                .clone()
                .record_failure("workspace_estimation", error)
        })?;
    diagnostics
        .clone()
        .with_required_bytes(workspace.required_bytes)
        .record_success("workspace_estimation", workspace.required_bytes);
    Ok(workspace)
}

pub fn query_workspace_raw(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims_override: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    let diagnostics = make_query_diagnostics(
        "raw.query_workspace",
        representation,
        shell_tuple,
        dims_override,
        options,
    );
    let (validated_inputs, validated_shape) = validate_raw_query_inputs(
        basis,
        operator,
        representation,
        shell_tuple,
        dims_override,
        options,
    )
    .map_err(|error| diagnostics.clone().record_failure("validation", error))?;
    let diagnostics = diagnostics.with_dims(validated_shape.dims.clone());
    let workspace =
        estimate_workspace(&validated_inputs, &validated_shape, options).map_err(|error| {
            diagnostics
                .clone()
                .record_failure("workspace_estimation", error)
        })?;
    diagnostics
        .clone()
        .with_required_bytes(workspace.required_bytes)
        .record_success("workspace_estimation", workspace.required_bytes);
    Ok(workspace)
}

pub fn estimate_workspace(
    validated_inputs: &ValidatedInputs,
    validated_shape: &ValidatedShape,
    options: &WorkspaceQueryOptions,
) -> Result<WorkspaceQuery, LibcintRsError> {
    let element_width_bytes = match validated_inputs.representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    };

    let scratch_bytes = compute_scratch_bytes(
        &validated_inputs.shell_angular_momentum,
        validated_inputs.primitive_count,
        validated_shape.dims.len(),
        validated_inputs.operator.kind(),
        validated_inputs.operator.family(),
        validated_inputs.feature_flags.len(),
    )?;
    let memory_plan = build_memory_plan(
        validated_shape.element_count,
        element_width_bytes,
        scratch_bytes,
        options.memory_limit_bytes,
    )?;

    Ok(WorkspaceQuery {
        required_bytes: memory_plan.required_bytes,
        alignment_bytes: super::memory::chunking::DEFAULT_ALIGNMENT_BYTES,
        natural_dims: validated_shape.natural_dims.clone(),
        dims: validated_shape.dims.clone(),
        element_count: validated_shape.element_count,
        scratch_bytes: memory_plan.scratch_bytes,
    })
}
