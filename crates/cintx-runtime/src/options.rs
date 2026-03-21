use tracing::Span;

pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
}

impl ExecutionOptions {
    pub fn with_default_limits() -> Self {
        Self {
            memory_limit_bytes: Some(DEFAULT_MEMORY_LIMIT_BYTES),
            ..Self::default()
        }
    }

    pub const fn default_memory_limit_bytes() -> usize {
        DEFAULT_MEMORY_LIMIT_BYTES
    }

    pub fn effective_memory_limit_bytes(&self, required_bytes: usize) -> usize {
        self.memory_limit_bytes.unwrap_or(required_bytes)
    }
}
