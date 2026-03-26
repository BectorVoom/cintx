use std::sync::Arc;

#[derive(Debug, Default)]
struct OptimizerMetadata {
    symbol_hint: Option<&'static str>,
    workspace_hint_bytes: Option<usize>,
}

/// Minimal compat-owned optimizer contract. Lifecycle APIs land in Plan 07.
#[derive(Clone, Debug, Default)]
pub struct RawOptimizerHandle {
    inner: Arc<OptimizerMetadata>,
}

impl RawOptimizerHandle {
    pub fn symbol_hint(&self) -> Option<&'static str> {
        self.inner.symbol_hint
    }

    pub fn workspace_hint_bytes(&self) -> Option<usize> {
        self.inner.workspace_hint_bytes
    }

    pub(crate) fn with_hints(
        symbol_hint: Option<&'static str>,
        workspace_hint_bytes: Option<usize>,
    ) -> Self {
        Self {
            inner: Arc::new(OptimizerMetadata {
                symbol_hint,
                workspace_hint_bytes,
            }),
        }
    }
}
