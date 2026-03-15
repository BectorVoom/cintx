use crate::contracts::{
    validate_dims, BasisSet, IntegralFamily, Operator, OperatorKind, Representation,
};
use crate::errors::LibcintRsError;

use super::{ExecutionRequest, WorkspaceQueryOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedExecution {
    pub request: ExecutionRequest,
    pub natural_dims: Vec<usize>,
    pub dims: Vec<usize>,
    pub element_count: usize,
    pub element_width_bytes: usize,
    pub required_output_bytes: usize,
}

impl PlannedExecution {
    pub fn required_elements(&self) -> usize {
        self.element_count
    }
}

pub fn plan_safe(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> Result<PlannedExecution, LibcintRsError> {
    plan_execution(
        basis,
        ExecutionRequest::from_safe(operator, representation, shell_tuple, options),
    )
}

pub fn plan_raw(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> Result<PlannedExecution, LibcintRsError> {
    plan_execution(
        basis,
        ExecutionRequest::from_raw(operator, representation, shell_tuple, dims, options),
    )
}

pub fn plan_execution(
    basis: &BasisSet,
    request: ExecutionRequest,
) -> Result<PlannedExecution, LibcintRsError> {
    let expected_arity = family_arity(request.operator.family);
    if request.shell_tuple.len() != expected_arity {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: expected_arity,
            got: request.shell_tuple.len(),
        });
    }

    let shells = basis.shells();
    let mut natural_dims = Vec::with_capacity(request.shell_tuple.len());
    for shell_index in &request.shell_tuple {
        let shell = shells
            .get(*shell_index)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell_tuple",
                reason: format!(
                    "index {shell_index} is out of bounds for {} shells",
                    shells.len()
                ),
            })?;
        natural_dims.push(shell_component_count(
            shell.angular_momentum(),
            request.representation,
        )?);
    }
    if let Some(extra_dim) = extra_component_dim(request.operator.family, request.operator.kind) {
        natural_dims.push(extra_dim);
    }

    let dims = match request.dims.as_deref() {
        Some(provided_dims) => {
            validate_dims(&natural_dims, provided_dims)?;
            provided_dims.to_vec()
        }
        None => natural_dims.clone(),
    };
    let element_count = checked_product(&dims)?;
    let element_width_bytes = representation_width_bytes(request.representation);
    let required_output_bytes =
        element_count
            .checked_mul(element_width_bytes)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "required output byte computation overflows usize".to_string(),
            })?;

    Ok(PlannedExecution {
        request,
        natural_dims,
        dims,
        element_count,
        element_width_bytes,
        required_output_bytes,
    })
}

fn checked_product(dims: &[usize]) -> Result<usize, LibcintRsError> {
    let mut product = 1usize;
    for dim in dims {
        if *dim == 0 {
            return Err(LibcintRsError::InvalidInput {
                field: "dims",
                reason: "dimension values must be greater than zero".to_string(),
            });
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "dims",
                reason: "dimension product overflows usize".to_string(),
            })?;
    }
    Ok(product)
}

fn shell_component_count(
    angular_momentum: u8,
    representation: Representation,
) -> Result<usize, LibcintRsError> {
    let l = usize::from(angular_momentum);
    // The safe shell model currently represents one contracted shell per entry, so nctr=1 here.
    match representation {
        Representation::Cartesian => cartesian_len(l),
        Representation::Spherical => spherical_len(l),
        Representation::Spinor => spinor_len(l, 0),
    }
}

fn cartesian_len(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    let l_plus_1 = angular_momentum
        .checked_add(1)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let l_plus_2 = angular_momentum
        .checked_add(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;

    let numerator = l_plus_1
        .checked_mul(l_plus_2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "cartesian component computation overflows usize".to_string(),
        })?;
    Ok(numerator / 2)
}

fn spherical_len(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    angular_momentum
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "spherical component computation overflows usize".to_string(),
        })
}

fn spinor_len(angular_momentum: usize, kappa: i32) -> Result<usize, LibcintRsError> {
    if kappa == 0 {
        return angular_momentum
            .checked_mul(4)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell.angular_momentum",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if kappa < 0 {
        return angular_momentum
            .checked_mul(2)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell.angular_momentum",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if angular_momentum == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "shell.kappa",
            reason: "positive kappa requires angular momentum > 0".to_string(),
        });
    }

    angular_momentum
        .checked_mul(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "spinor component computation overflows usize".to_string(),
        })
}

fn representation_width_bytes(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    }
}

fn family_arity(family: IntegralFamily) -> usize {
    match family {
        IntegralFamily::OneElectron | IntegralFamily::TwoCenterTwoElectron => 2,
        IntegralFamily::ThreeCenterOneElectron | IntegralFamily::ThreeCenterTwoElectron => 3,
        IntegralFamily::TwoElectron => 4,
    }
}

fn extra_component_dim(family: IntegralFamily, kind: OperatorKind) -> Option<usize> {
    match (family, kind) {
        (IntegralFamily::ThreeCenterTwoElectron, OperatorKind::ElectronRepulsion) => Some(3),
        _ => None,
    }
}
