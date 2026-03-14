use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::contracts::Representation;
use crate::errors::LibcintRsError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryDiagnostics {
    pub correlation_id: u64,
    pub api: &'static str,
    pub representation: &'static str,
    pub shell_tuple: Vec<usize>,
    pub dims: Vec<usize>,
    pub required_bytes: Option<usize>,
    pub provided_bytes: Option<usize>,
    pub memory_limit_bytes: Option<usize>,
    pub backend_candidate: &'static str,
    pub feature_flags: Vec<&'static str>,
}

impl QueryDiagnostics {
    pub fn new(
        api: &'static str,
        representation: Representation,
        shell_tuple: Vec<usize>,
        dims: Vec<usize>,
        memory_limit_bytes: Option<usize>,
        backend_candidate: &'static str,
        feature_flags: Vec<&'static str>,
    ) -> Self {
        let mut diagnostics = Self {
            correlation_id: 0,
            api,
            representation: representation.as_str(),
            shell_tuple,
            dims,
            required_bytes: None,
            provided_bytes: None,
            memory_limit_bytes,
            backend_candidate,
            feature_flags,
        };
        diagnostics.refresh_correlation_id();
        diagnostics
    }

    pub fn with_dims(mut self, dims: Vec<usize>) -> Self {
        self.dims = dims;
        self.provided_bytes = dims_to_bytes(&self.dims, self.representation);
        self.refresh_correlation_id();
        self
    }

    pub fn with_required_bytes(mut self, required_bytes: usize) -> Self {
        self.required_bytes = Some(required_bytes);
        self.refresh_correlation_id();
        self
    }

    pub fn with_provided_bytes(mut self, provided_bytes: usize) -> Self {
        self.provided_bytes = Some(provided_bytes);
        self.refresh_correlation_id();
        self
    }

    pub fn with_provided_bytes_from_dims(mut self) -> Self {
        self.provided_bytes = dims_to_bytes(&self.dims, self.representation);
        self.refresh_correlation_id();
        self
    }

    pub fn record_failure(mut self, stage: &'static str, error: LibcintRsError) -> Box<QueryError> {
        match &error {
            LibcintRsError::DimsBufferMismatch { provided, .. } => {
                self.dims = provided.clone();
            }
            LibcintRsError::MemoryLimitExceeded {
                required_bytes,
                limit_bytes,
            } => {
                self.required_bytes = Some(*required_bytes);
                self.memory_limit_bytes = Some(*limit_bytes);
            }
            _ => {}
        }

        self = self.with_provided_bytes_from_dims();
        tracing::error!(
            correlation_id = self.correlation_id,
            stage,
            api = self.api,
            representation = self.representation,
            shell_tuple = ?self.shell_tuple,
            dims = ?self.dims,
            required_bytes = ?self.required_bytes,
            provided_bytes = ?self.provided_bytes,
            memory_limit_bytes = ?self.memory_limit_bytes,
            backend_candidate = self.backend_candidate,
            feature_flags = ?self.feature_flags,
            error = ?error,
            "query_workspace_failure"
        );

        Box::new(QueryError {
            error,
            diagnostics: self,
        })
    }

    pub fn record_success(&self, stage: &'static str, required_bytes: usize) {
        tracing::debug!(
            correlation_id = self.correlation_id,
            stage,
            api = self.api,
            representation = self.representation,
            shell_tuple = ?self.shell_tuple,
            dims = ?self.dims,
            required_bytes,
            memory_limit_bytes = ?self.memory_limit_bytes,
            backend_candidate = self.backend_candidate,
            feature_flags = ?self.feature_flags,
            "query_workspace_success"
        );
    }

    fn refresh_correlation_id(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.api.hash(&mut hasher);
        self.representation.hash(&mut hasher);
        self.shell_tuple.hash(&mut hasher);
        self.dims.hash(&mut hasher);
        self.required_bytes.hash(&mut hasher);
        self.provided_bytes.hash(&mut hasher);
        self.memory_limit_bytes.hash(&mut hasher);
        self.backend_candidate.hash(&mut hasher);
        self.feature_flags.hash(&mut hasher);
        self.correlation_id = hasher.finish();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{error}")]
pub struct QueryError {
    pub error: LibcintRsError,
    pub diagnostics: QueryDiagnostics,
}

pub type QueryResult<T> = Result<T, Box<QueryError>>;

fn dims_to_bytes(dims: &[usize], representation: &str) -> Option<usize> {
    if dims.is_empty() {
        return None;
    }

    let element_width_bytes = if representation == Representation::Spinor.as_str() {
        16usize
    } else {
        8usize
    };

    let mut elements = 1usize;
    for dim in dims {
        if *dim == 0 {
            return None;
        }
        elements = elements.checked_mul(*dim)?;
    }

    elements.checked_mul(element_width_bytes)
}
