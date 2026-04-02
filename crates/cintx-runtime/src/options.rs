use tracing::Span;

pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Selects which CubeCL compute backend to use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// wgpu-based GPU backend (default).
    Wgpu,
    /// CPU backend for oracle/test use and environments without GPU.
    Cpu,
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::Wgpu
    }
}

/// Runtime backend selection intent carried via execution options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendIntent {
    /// Which backend kind to use.
    pub backend: BackendKind,
    /// Device selector string (e.g. "auto", "discrete:0", "integrated:0").
    pub selector: String,
}

impl Default for BackendIntent {
    fn default() -> Self {
        Self {
            backend: BackendKind::Wgpu,
            selector: "auto".to_owned(),
        }
    }
}

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
