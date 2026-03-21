use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::QueryResult;
use crate::runtime::validator::make_query_diagnostics;
use crate::runtime::{
    EvaluationMetadata, EvaluationOutput, EvaluationOutputMut, EvaluationTensor, WorkspaceQuery,
    WorkspaceQueryOptions, evaluate as runtime_evaluate, evaluate_into as runtime_evaluate_into,
    query_workspace_safe,
};

pub use crate::runtime::{
    EvaluationMetadata as SafeEvaluationMetadata, EvaluationOutput as SafeEvaluationOutput,
    EvaluationOutputMut as SafeEvaluationOutputMut, EvaluationTensor as SafeEvaluationTensor,
};

pub fn query_workspace(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    query_workspace_safe(basis, operator, representation, shell_tuple, options)
}

pub fn evaluate_into(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
    output: EvaluationOutputMut<'_>,
) -> QueryResult<EvaluationMetadata> {
    let mut diagnostics = make_query_diagnostics(
        "safe.evaluate_into",
        representation,
        shell_tuple,
        None,
        options,
    );
    if let Some(provided_bytes) = output.provided_bytes(representation) {
        diagnostics = diagnostics.with_provided_bytes(provided_bytes);
    }

    let metadata = runtime_evaluate_into(
        basis,
        operator,
        representation,
        shell_tuple,
        options,
        output,
    )
    .map_err(|error| diagnostics.clone().record_failure("execution", error))?;

    diagnostics
        .clone()
        .with_dims(metadata.dims.clone())
        .with_required_bytes(metadata.required_bytes)
        .record_success("execution", metadata.required_bytes);
    Ok(metadata)
}

pub fn evaluate(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> QueryResult<EvaluationTensor> {
    let diagnostics =
        make_query_diagnostics("safe.evaluate", representation, shell_tuple, None, options);
    let tensor = runtime_evaluate(basis, operator, representation, shell_tuple, options)
        .map_err(|error| diagnostics.clone().record_failure("execution", error))?;
    let required_bytes = output_required_bytes(&tensor.output).ok_or_else(|| {
        diagnostics.clone().record_failure(
            "execution",
            crate::errors::LibcintRsError::InvalidInput {
                field: "output",
                reason: "required output byte computation overflows usize".to_string(),
            },
        )
    })?;
    diagnostics
        .clone()
        .with_dims(tensor.dims.clone())
        .with_required_bytes(required_bytes)
        .record_success("execution", required_bytes);
    Ok(tensor)
}

fn output_required_bytes(output: &EvaluationOutput) -> Option<usize> {
    match output {
        EvaluationOutput::Real(values) => values.len().checked_mul(8),
        EvaluationOutput::Spinor(values) => values.len().checked_mul(16),
    }
}
