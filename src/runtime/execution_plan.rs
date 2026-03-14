use crate::contracts::{IntegralFamily, Operator, OperatorKind, Representation};

use super::WorkspaceQueryOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionOperator {
    pub family: IntegralFamily,
    pub kind: OperatorKind,
}

impl From<Operator> for ExecutionOperator {
    fn from(operator: Operator) -> Self {
        Self {
            family: operator.family(),
            kind: operator.kind(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionMemoryOptions {
    pub memory_limit_bytes: Option<usize>,
    pub backend_candidate: String,
    pub feature_flags: Vec<String>,
}

impl From<&WorkspaceQueryOptions> for ExecutionMemoryOptions {
    fn from(options: &WorkspaceQueryOptions) -> Self {
        Self {
            memory_limit_bytes: options.memory_limit_bytes,
            backend_candidate: options.backend_candidate.to_string(),
            feature_flags: options
                .feature_flags
                .iter()
                .map(|flag| (*flag).to_string())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRequest {
    pub operator: ExecutionOperator,
    pub representation: Representation,
    pub shell_tuple: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub memory: ExecutionMemoryOptions,
}

impl ExecutionRequest {
    pub fn from_safe(
        operator: Operator,
        representation: Representation,
        shell_tuple: &[usize],
        options: &WorkspaceQueryOptions,
    ) -> Self {
        Self::new(operator, representation, shell_tuple, None, options)
    }

    pub fn from_raw(
        operator: Operator,
        representation: Representation,
        shell_tuple: &[usize],
        dims: Option<&[usize]>,
        options: &WorkspaceQueryOptions,
    ) -> Self {
        Self::new(operator, representation, shell_tuple, dims, options)
    }

    pub fn new(
        operator: Operator,
        representation: Representation,
        shell_tuple: &[usize],
        dims: Option<&[usize]>,
        options: &WorkspaceQueryOptions,
    ) -> Self {
        Self {
            operator: ExecutionOperator::from(operator),
            representation,
            shell_tuple: shell_tuple.to_vec(),
            dims: dims.map(<[usize]>::to_vec),
            memory: ExecutionMemoryOptions::from(options),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionBackend {
    CpuReference,
}

impl ExecutionBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuReference => "cpu-reference",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionDispatch {
    pub backend: ExecutionBackend,
    pub request: ExecutionRequest,
}

impl ExecutionDispatch {
    pub fn cpu(request: ExecutionRequest) -> Self {
        Self {
            backend: ExecutionBackend::CpuReference,
            request,
        }
    }
}
