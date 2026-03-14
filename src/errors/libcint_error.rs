#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LibcintRsError {
    #[error("unsupported API `{api}`: {reason}")]
    UnsupportedApi {
        api: &'static str,
        reason: &'static str,
    },
    #[error("unsupported representation `{representation}` for API `{api}`")]
    UnsupportedRepresentation {
        api: &'static str,
        representation: &'static str,
    },
    #[error("invalid input for {field}: {reason}")]
    InvalidInput { field: &'static str, reason: String },
    #[error("invalid layout for {item}: expected {expected}, got {got}")]
    InvalidLayout {
        item: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("dims/buffer mismatch: expected {expected:?}, provided {provided:?}")]
    DimsBufferMismatch {
        expected: Vec<usize>,
        provided: Vec<usize>,
    },
    #[error("memory limit exceeded: required {required_bytes} bytes, limit {limit_bytes} bytes")]
    MemoryLimitExceeded {
        required_bytes: usize,
        limit_bytes: usize,
    },
    #[error("allocation failure in `{operation}`: {detail}")]
    AllocationFailure {
        operation: &'static str,
        detail: String,
    },
    #[error("backend failure in `{backend}`: {detail}")]
    BackendFailure {
        backend: &'static str,
        detail: String,
    },
}
