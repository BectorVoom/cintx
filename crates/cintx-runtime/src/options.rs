use tracing::Span;

pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Which compute backend to target for integral evaluation.
///
/// Per D-03, backend kind is control-plane metadata carried in `ExecutionOptions` and
/// propagated through `WorkspaceQuery` so query/evaluate drift can be detected and
/// rejected with a typed error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// wgpu-backed CubeCL runtime (the primary production backend).
    Wgpu,
    /// CPU execution profile, used for testing and oracle comparison only.
    Cpu,
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::Wgpu
    }
}

/// Backend selection intent carried through the query/evaluate contract.
///
/// `selector` is an advisory hint that lets callers express device preference
/// (e.g. `"auto"`, `"device:0"`).  The runtime MAY use it for adapter selection
/// but must fail closed if the requested adapter is unavailable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendIntent {
    pub backend: BackendKind,
    /// Adapter selection hint.  `"auto"` means the runtime picks the best
    /// available wgpu adapter.
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

/// A snapshot of the concrete backend adapter capabilities at query time.
///
/// When `query_workspace` runs, the planner records the active adapter's
/// identity and a lightweight fingerprint of its capabilities.  `evaluate`
/// then rejects the plan if the token has drifted (e.g. if the caller swapped
/// adapters between query and evaluate), satisfying D-08.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendCapabilityToken {
    /// Human-readable adapter identifier (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub adapter_name: String,
    /// Backend API string (e.g. `"wgpu"`, `"cpu"`).
    pub backend_api: String,
    /// Lightweight capability fingerprint.  Must change whenever device
    /// features or limits relevant to execution differ.
    pub capability_fingerprint: u64,
}

impl Default for BackendCapabilityToken {
    fn default() -> Self {
        Self {
            adapter_name: String::new(),
            backend_api: "wgpu".to_owned(),
            capability_fingerprint: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
    /// Backend selection intent that must remain stable across query/evaluate.
    pub backend_intent: BackendIntent,
    /// Capability token snapshotted at query time; compared at evaluate time.
    pub backend_capability_token: BackendCapabilityToken,
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
