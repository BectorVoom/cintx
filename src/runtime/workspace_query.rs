use crate::contracts::{BasisSet, IntegralFamily, Operator, OperatorKind, Representation};
use crate::diagnostics::QueryResult;
use crate::errors::LibcintRsError;

use super::validator::{
    ValidatedInputs, ValidatedShape, WorkspaceQueryOptions, make_query_diagnostics,
    validate_raw_query_inputs, validate_safe_query_inputs,
};

const DEFAULT_ALIGNMENT_BYTES: usize = 64;

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
    let diagnostics = diagnostics
        .with_dims(validated_shape.dims.clone())
        .with_provided_bytes_from_dims();
    let workspace = estimate_workspace(&validated_inputs, &validated_shape, options)
        .map_err(|error| diagnostics.clone().record_failure("workspace_estimation", error))?;
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
    )
    .with_provided_bytes_from_dims();
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
    let workspace = estimate_workspace(&validated_inputs, &validated_shape, options)
        .map_err(|error| diagnostics.clone().record_failure("workspace_estimation", error))?;
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

    let payload_bytes = validated_shape
        .element_count
        .checked_mul(element_width_bytes)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "payload byte computation overflows usize".to_string(),
        })?;

    let angular_complexity = validated_inputs
        .shell_angular_momentum
        .iter()
        .fold(0usize, |acc, ang| acc + usize::from(*ang) + 1);
    let operator_scale = operator_scale(validated_inputs.operator.kind());
    let family_scale = family_scale(validated_inputs.operator.family());
    let feature_scale = validated_inputs.feature_flags.len().max(1);
    let scratch_units = angular_complexity
        .checked_add(validated_inputs.primitive_count)
        .and_then(|value| value.checked_add(validated_shape.dims.len()))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "scratch unit computation overflows usize".to_string(),
        })?;
    let scratch_bytes = scratch_units
        .checked_mul(16)
        .and_then(|value| value.checked_mul(operator_scale))
        .and_then(|value| value.checked_mul(family_scale))
        .and_then(|value| value.checked_mul(feature_scale))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "scratch byte computation overflows usize".to_string(),
        })?;

    let required_bytes = payload_bytes
        .checked_add(scratch_bytes)
        .and_then(|bytes| align_up(bytes, DEFAULT_ALIGNMENT_BYTES))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "required byte computation overflows usize".to_string(),
        })?;

    if let Some(limit_bytes) = options.memory_limit_bytes {
        if required_bytes > limit_bytes {
            return Err(LibcintRsError::MemoryLimitExceeded {
                required_bytes,
                limit_bytes,
            });
        }
    }

    Ok(WorkspaceQuery {
        required_bytes,
        alignment_bytes: DEFAULT_ALIGNMENT_BYTES,
        natural_dims: validated_shape.natural_dims.clone(),
        dims: validated_shape.dims.clone(),
        element_count: validated_shape.element_count,
        scratch_bytes,
    })
}

fn operator_scale(kind: OperatorKind) -> usize {
    match kind {
        OperatorKind::Overlap => 1,
        OperatorKind::Kinetic => 2,
        OperatorKind::NuclearAttraction => 3,
        OperatorKind::ElectronRepulsion => 4,
    }
}

fn family_scale(family: IntegralFamily) -> usize {
    match family {
        IntegralFamily::OneElectron => 2,
        IntegralFamily::TwoCenterTwoElectron => 2,
        IntegralFamily::ThreeCenterOneElectron => 3,
        IntegralFamily::ThreeCenterTwoElectron => 3,
        IntegralFamily::TwoElectron => 4,
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }

    value.checked_add(alignment - 1).map(|v| v & !(alignment - 1))
}
