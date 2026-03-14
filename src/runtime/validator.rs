use crate::contracts::{BasisSet, Operator, Representation, validate_dims};
use crate::errors::LibcintRsError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceQueryOptions {
    pub memory_limit_bytes: Option<usize>,
    pub backend_candidate: &'static str,
    pub feature_flags: Vec<&'static str>,
}

impl Default for WorkspaceQueryOptions {
    fn default() -> Self {
        Self {
            memory_limit_bytes: None,
            backend_candidate: "cpu",
            feature_flags: Vec::new(),
        }
    }
}

impl WorkspaceQueryOptions {
    pub fn normalized_feature_flags(&self) -> Vec<&'static str> {
        let mut flags = self.feature_flags.clone();
        flags.sort_unstable();
        flags.dedup();
        flags
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedInputs {
    pub api: &'static str,
    pub operator: Operator,
    pub representation: Representation,
    pub shell_tuple: Vec<usize>,
    pub shell_angular_momentum: Vec<u8>,
    pub primitive_count: usize,
    pub backend_candidate: &'static str,
    pub feature_flags: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedShape {
    pub natural_dims: Vec<usize>,
    pub dims: Vec<usize>,
    pub element_count: usize,
}

pub fn validate_safe_query_inputs(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> Result<(ValidatedInputs, ValidatedShape), LibcintRsError> {
    validate_query_inputs(
        "safe.query_workspace",
        basis,
        operator,
        representation,
        shell_tuple,
        None,
        options,
    )
}

pub fn validate_raw_query_inputs(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims_override: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> Result<(ValidatedInputs, ValidatedShape), LibcintRsError> {
    validate_query_inputs(
        "raw.query_workspace",
        basis,
        operator,
        representation,
        shell_tuple,
        dims_override,
        options,
    )
}

fn validate_query_inputs(
    api: &'static str,
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims_override: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> Result<(ValidatedInputs, ValidatedShape), LibcintRsError> {
    let expected_arity = family_arity(operator.family());
    if shell_tuple.len() != expected_arity {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: expected_arity,
            got: shell_tuple.len(),
        });
    }

    let shells = basis.shells();
    let mut natural_dims = Vec::with_capacity(shell_tuple.len());
    let mut shell_angular_momentum = Vec::with_capacity(shell_tuple.len());
    let mut primitive_count = 0usize;

    for shell_index in shell_tuple {
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
        primitive_count += shell.primitives().len();
        natural_dims.push(component_count(shell.angular_momentum(), representation));
    }

    let dims = match dims_override {
        Some(provided_dims) => {
            validate_dims(&natural_dims, provided_dims)?;
            provided_dims.to_vec()
        }
        None => natural_dims.clone(),
    };

    let element_count = checked_product(&dims)?;
    let validated_inputs = ValidatedInputs {
        api,
        operator,
        representation,
        shell_tuple: shell_tuple.to_vec(),
        shell_angular_momentum,
        primitive_count,
        backend_candidate: options.backend_candidate,
        feature_flags: options.normalized_feature_flags(),
    };
    let validated_shape = ValidatedShape {
        natural_dims,
        dims,
        element_count,
    };

    Ok((validated_inputs, validated_shape))
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

fn family_arity(family: crate::contracts::IntegralFamily) -> usize {
    use crate::contracts::IntegralFamily;

    match family {
        IntegralFamily::OneElectron => 2,
        IntegralFamily::TwoElectron => 4,
        IntegralFamily::TwoCenterTwoElectron => 2,
        IntegralFamily::ThreeCenterOneElectron => 3,
        IntegralFamily::ThreeCenterTwoElectron => 3,
    }
}

fn component_count(angular_momentum: u8, representation: Representation) -> usize {
    let l = usize::from(angular_momentum);
    match representation {
        Representation::Cartesian => ((l + 1) * (l + 2)) / 2,
        Representation::Spherical => (2 * l) + 1,
        Representation::Spinor => 2 * ((2 * l) + 1),
    }
}
