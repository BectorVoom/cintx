use crate::errors::LibcintRsError;

use super::{EvaluationOutputMut, LayoutElementKind, OutputLayout};

#[derive(Debug)]
pub struct OutputWriter<'a> {
    output: EvaluationOutputMut<'a>,
    staged: StagedOutput,
}

#[derive(Debug)]
pub enum StagedOutputMut<'a> {
    Real(&'a mut [f64]),
    Spinor(&'a mut [[f64; 2]]),
}

#[derive(Debug)]
enum StagedOutput {
    Real(Vec<f64>),
    Spinor(Vec<[f64; 2]>),
}

impl<'a> OutputWriter<'a> {
    pub fn new(layout: &OutputLayout, output: EvaluationOutputMut<'a>) -> Result<Self, LibcintRsError> {
        validate_output_contract(layout, &output)?;
        let staged = allocate_staged(layout)?;
        Ok(Self { output, staged })
    }

    pub fn staged_output_mut(&mut self) -> StagedOutputMut<'_> {
        match &mut self.staged {
            StagedOutput::Real(values) => StagedOutputMut::Real(values.as_mut_slice()),
            StagedOutput::Spinor(values) => StagedOutputMut::Spinor(values.as_mut_slice()),
        }
    }

    pub fn commit(self) -> Result<(), LibcintRsError> {
        match (self.output, self.staged) {
            (EvaluationOutputMut::Real(output), StagedOutput::Real(staged)) => {
                output.copy_from_slice(&staged);
                Ok(())
            }
            (EvaluationOutputMut::Spinor(output), StagedOutput::Spinor(staged)) => {
                output.copy_from_slice(&staged);
                Ok(())
            }
            _ => Err(LibcintRsError::BackendFailure {
                backend: "cpu-reference",
                detail: "output staging contract drifted during commit".to_string(),
            }),
        }
    }
}

fn validate_output_contract(
    layout: &OutputLayout,
    output: &EvaluationOutputMut<'_>,
) -> Result<(), LibcintRsError> {
    match (layout.element_kind, output) {
        (LayoutElementKind::RealF64, EvaluationOutputMut::Real(values)) => {
            layout.validate_real_buffer_len(values.len())
        }
        (LayoutElementKind::ComplexF64Pair, EvaluationOutputMut::Spinor(values)) => {
            layout.validate_complex_buffer_len(values.len())
        }
        _ => Err(LibcintRsError::UnsupportedRepresentation {
            api: "output.writer",
            representation: layout.representation.as_str(),
        }),
    }
}

fn allocate_staged(layout: &OutputLayout) -> Result<StagedOutput, LibcintRsError> {
    match layout.element_kind {
        LayoutElementKind::RealF64 => {
            let mut staged = Vec::new();
            staged.try_reserve_exact(layout.element_count).map_err(|_| {
                LibcintRsError::AllocationFailure {
                    operation: "output_writer.real_staging",
                    detail: format!("failed to reserve {} f64 elements", layout.element_count),
                }
            })?;
            staged.resize(layout.element_count, 0.0);
            Ok(StagedOutput::Real(staged))
        }
        LayoutElementKind::ComplexF64Pair => {
            let mut staged = Vec::new();
            staged.try_reserve_exact(layout.element_count).map_err(|_| {
                LibcintRsError::AllocationFailure {
                    operation: "output_writer.spinor_staging",
                    detail: format!(
                        "failed to reserve {} spinor complex elements",
                        layout.element_count
                    ),
                }
            })?;
            staged.resize(layout.element_count, [0.0, 0.0]);
            Ok(StagedOutput::Spinor(staged))
        }
    }
}
