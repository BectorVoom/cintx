//! Stable facade-level error categories for the safe API.

use cintx_core::{Representation, cintxRsError};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FacadeErrorKind {
    UnsupportedApi,
    Layout,
    Memory,
    Validation,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FacadeError {
    #[error("unsupported api: {requested}")]
    UnsupportedApi { requested: String },
    #[error("layout contract violation: {detail}")]
    Layout { detail: String },
    #[error("memory contract violation: {detail}")]
    Memory { detail: String },
    #[error("validation failed: {detail}")]
    Validation { detail: String },
}

impl FacadeError {
    pub const fn kind(&self) -> FacadeErrorKind {
        match self {
            Self::UnsupportedApi { .. } => FacadeErrorKind::UnsupportedApi,
            Self::Layout { .. } => FacadeErrorKind::Layout,
            Self::Memory { .. } => FacadeErrorKind::Memory,
            Self::Validation { .. } => FacadeErrorKind::Validation,
        }
    }

    fn unsupported_representation(operator: String, representation: Representation) -> Self {
        Self::UnsupportedApi {
            requested: format!(
                "{operator} does not support representation {representation}"
            ),
        }
    }
}

impl From<cintxRsError> for FacadeError {
    fn from(value: cintxRsError) -> Self {
        match value {
            cintxRsError::UnsupportedApi { requested } => Self::UnsupportedApi { requested },
            cintxRsError::UnsupportedRepresentation {
                operator,
                representation,
            } => Self::unsupported_representation(operator, representation),
            cintxRsError::InvalidDims { expected, provided } => Self::Layout {
                detail: format!("InvalidDims expected={expected} provided={provided}"),
            },
            cintxRsError::InvalidAtmLayout {
                slot_width,
                provided,
            } => Self::Layout {
                detail: format!("InvalidAtmLayout slot_width={slot_width} provided={provided}"),
            },
            cintxRsError::InvalidBasLayout {
                slot_width,
                provided,
            } => Self::Layout {
                detail: format!("InvalidBasLayout slot_width={slot_width} provided={provided}"),
            },
            cintxRsError::InvalidEnvOffset {
                slot,
                offset,
                env_len,
            } => Self::Layout {
                detail: format!("InvalidEnvOffset slot={slot} offset={offset} env_len={env_len}"),
            },
            cintxRsError::BufferTooSmall { required, provided } => Self::Layout {
                detail: format!("BufferTooSmall required={required} provided={provided}"),
            },
            cintxRsError::MemoryLimitExceeded { requested, limit } => Self::Memory {
                detail: format!("MemoryLimitExceeded requested={requested} limit={limit}"),
            },
            cintxRsError::HostAllocationFailed { bytes } => Self::Memory {
                detail: format!("HostAllocationFailed bytes={bytes}"),
            },
            cintxRsError::DeviceOutOfMemory { bytes, device } => Self::Memory {
                detail: format!("DeviceOutOfMemory bytes={bytes} device={device}"),
            },
            cintxRsError::InvalidShellTuple { expected, got } => Self::Validation {
                detail: format!("InvalidShellTuple expected={expected} got={got}"),
            },
            cintxRsError::InvalidShellAtomIndex { index, atom_count } => Self::Validation {
                detail: format!("InvalidShellAtomIndex index={index} atom_count={atom_count}"),
            },
            cintxRsError::ChunkPlanFailed { from, detail } => Self::Validation {
                detail: format!("ChunkPlanFailed from={from} detail={detail}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FacadeError, FacadeErrorKind};
    use cintx_core::cintxRsError;

    #[test]
    fn memory_limit_maps_to_memory_kind() {
        let err = FacadeError::from(cintxRsError::MemoryLimitExceeded {
            requested: 1024,
            limit: 512,
        });

        assert_eq!(err.kind(), FacadeErrorKind::Memory);
        assert!(matches!(err, FacadeError::Memory { .. }));
    }

    #[test]
    fn invalid_dims_maps_to_layout_kind() {
        let err = FacadeError::from(cintxRsError::InvalidDims {
            expected: 4,
            provided: 2,
        });

        assert_eq!(err.kind(), FacadeErrorKind::Layout);
        assert!(matches!(err, FacadeError::Layout { .. }));
    }
}
