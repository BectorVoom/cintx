//! Driver error surface.

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("integral evaluation failed for shells {shells:?}: {detail}")]
    Evaluation { shells: [i32; 4], detail: String },

    #[error("output buffer too small: need {required}, have {provided}")]
    BufferTooSmall { required: usize, provided: usize },

    #[error(transparent)]
    Core(#[from] cintx_core::cintxRsError),
}
