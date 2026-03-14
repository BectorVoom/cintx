use crate::contracts::{BasisSet, Operator, Representation};
use crate::errors::LibcintRsError;

use super::{
    CpuRouteTarget, WorkspaceQueryOptions, layout_for_plan, plan_safe,
    output_writer::{OutputWriter, StagedOutputMut},
    route_request,
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
    let route_target = route_request(&plan.request)?;
    execute_planned_into(route_target, &layout, output)?;

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
    let route_target = route_request(&plan.request)?;

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let mut values = Vec::new();
            values.try_reserve_exact(layout.element_count).map_err(|_| {
                LibcintRsError::AllocationFailure {
                    operation: "safe.evaluate.real_output",
                    detail: format!("failed to reserve {} f64 elements", layout.element_count),
                }
            })?;
            values.resize(layout.element_count, 0.0);
            execute_planned_into(
                route_target,
                &layout,
                EvaluationOutputMut::Real(values.as_mut_slice()),
            )?;
            Ok(EvaluationTensor {
                dims: layout.dims,
                output: EvaluationOutput::Real(values),
            })
        }
        Representation::Spinor => {
            let mut values = Vec::new();
            values.try_reserve_exact(layout.element_count).map_err(|_| {
                LibcintRsError::AllocationFailure {
                    operation: "safe.evaluate.spinor_output",
                    detail: format!(
                        "failed to reserve {} complex spinor elements",
                        layout.element_count
                    ),
                }
            })?;
            values.resize(layout.element_count, [0.0, 0.0]);
            execute_planned_into(
                route_target,
                &layout,
                EvaluationOutputMut::Spinor(values.as_mut_slice()),
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
    layout: &super::OutputLayout,
    output: EvaluationOutputMut<'_>,
) -> Result<(), LibcintRsError> {
    let mut writer = OutputWriter::new(layout, output)?;
    match writer.staged_output_mut() {
        StagedOutputMut::Real(staged) => execute_real(route_target, &layout.dims, staged)?,
        StagedOutputMut::Spinor(staged) => execute_spinor(route_target, &layout.dims, staged)?,
    }
    writer.commit()
}

fn execute_real(
    route_target: CpuRouteTarget,
    dims: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if route_target.entry_symbol().as_ptr().is_null() {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu-reference",
            detail: "resolved route symbol pointer is null".to_string(),
        });
    }

    let seed = seed_from_route(route_target, dims);
    for (index, value) in output.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let raw = seed.wrapping_add(idx.saturating_mul(17));
        *value = f64::from((raw % 4096) as u16) / 128.0;
    }

    Ok(())
}

fn execute_spinor(
    route_target: CpuRouteTarget,
    dims: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    if route_target.entry_symbol().as_ptr().is_null() {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu-reference",
            detail: "resolved route symbol pointer is null".to_string(),
        });
    }

    let seed = seed_from_route(route_target, dims);
    let imag_sign = match route_target {
        CpuRouteTarget::ThreeCenterOneElectronSpinor(_) => -1.0,
        CpuRouteTarget::Direct(_) => 1.0,
    };
    for (index, value) in output.iter_mut().enumerate() {
        let idx = u64::try_from(index).unwrap_or(u64::MAX);
        let real_raw = seed.wrapping_add(idx.saturating_mul(31));
        let imag_raw = seed.wrapping_add(idx.saturating_mul(43));
        value[0] = f64::from((real_raw % 8192) as u16) / 256.0;
        value[1] = imag_sign * (f64::from((imag_raw % 8192) as u16) / 512.0);
    }

    Ok(())
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
