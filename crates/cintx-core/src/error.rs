use crate::operator::Representation;
use thiserror::Error;

/// Common errors for cintx-core domain constructors.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("atomic number must be between 1 and 118, got {0}")]
    InvalidAtomicNumber(u16),
    #[error("coordinate values must be finite, got {0:?}")]
    InvalidCoordinate([f64; 3]),
    #[error("extra nuclear data is not finite or negative")]
    InvalidNuclearDetail,
    #[error("fractional charge {0} is out of range [-2.0, 2.0]")]
    InvalidFractionalCharge(f64),
    #[error("at least one atom/ shell is required")]
    EmptyBasis,
    #[error("shell primitive count mismatch: expected {expected}, got {actual} for {field}")]
    ShellPrimitiveMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("shell primitive/ contraction counts must be positive, got nprim={nprim}, nctr={nctr}")]
    InvalidShellCounts { nprim: usize, nctr: usize },
    #[error("basis shell refers to missing atom index {index} (atoms={total})")]
    MissingAtomIndex { index: usize, total: usize },
    #[error("requested shell index {index} exceeds {total}")]
    ShellIndexOutOfBounds { index: usize, total: usize },
    #[error("shell tuple cannot exceed {limit} entries")]
    ShellTupleArityExceeded { limit: usize },
}

#[allow(non_camel_case_types)]
#[derive(Debug, Error)]
pub enum cintxRsError {
    #[error("unsupported api requested={requested}")]
    UnsupportedApi { requested: String },
    #[error("unsupported representation {representation} for {operator}")]
    UnsupportedRepresentation {
        operator: String,
        representation: Representation,
    },
    #[error("invalid shell tuple: expected {expected}, got {got}")]
    InvalidShellTuple { expected: usize, got: usize },
    #[error("invalid shell atom index {index}; basis has {atom_count} atoms")]
    InvalidShellAtomIndex { index: usize, atom_count: usize },
    #[error("invalid dims: expected {expected}, provided {provided}")]
    InvalidDims { expected: usize, provided: usize },
    #[error("invalid atm layout: slot_width={slot_width}, provided={provided}")]
    InvalidAtmLayout { slot_width: usize, provided: usize },
    #[error("invalid bas layout: slot_width={slot_width}, provided={provided}")]
    InvalidBasLayout { slot_width: usize, provided: usize },
    #[error("invalid env offset for {slot}: offset={offset}, env_len={env_len}")]
    InvalidEnvOffset {
        slot: &'static str,
        offset: usize,
        env_len: usize,
    },
    #[error("buffer too small: required={required}, provided={provided}")]
    BufferTooSmall { required: usize, provided: usize },
    #[error("memory limit exceeded: requested={requested}, limit={limit}")]
    MemoryLimitExceeded { requested: usize, limit: usize },
    #[error("host allocation failed for {bytes} bytes")]
    HostAllocationFailed { bytes: usize },
    #[error("device out of memory for {bytes} bytes on {device}")]
    DeviceOutOfMemory { bytes: usize, device: String },
    #[error("chunk plan failed in {from}: {detail}")]
    ChunkPlanFailed { from: &'static str, detail: String },
    #[error("invalid env parameter {param}: {reason}")]
    InvalidEnvParam { param: &'static str, reason: String },
}

pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::cintxRsError;

    #[test]
    fn invalid_env_param_formats_and_matches() {
        let err = cintxRsError::InvalidEnvParam {
            param: "PTR_F12_ZETA",
            reason: "must be non-zero".to_owned(),
        };
        assert!(matches!(
            err,
            cintxRsError::InvalidEnvParam {
                param: "PTR_F12_ZETA",
                ..
            }
        ));
        assert_eq!(
            err.to_string(),
            "invalid env parameter PTR_F12_ZETA: must be non-zero"
        );
    }

    #[test]
    fn invalid_atm_layout_formats_and_matches() {
        let err = cintxRsError::InvalidAtmLayout {
            slot_width: 6,
            provided: 10,
        };
        assert!(matches!(
            err,
            cintxRsError::InvalidAtmLayout {
                slot_width: 6,
                provided: 10
            }
        ));
        assert_eq!(
            err.to_string(),
            "invalid atm layout: slot_width=6, provided=10"
        );
    }

    #[test]
    fn invalid_bas_layout_formats_and_matches() {
        let err = cintxRsError::InvalidBasLayout {
            slot_width: 8,
            provided: 15,
        };
        assert!(matches!(
            err,
            cintxRsError::InvalidBasLayout {
                slot_width: 8,
                provided: 15
            }
        ));
        assert_eq!(
            err.to_string(),
            "invalid bas layout: slot_width=8, provided=15"
        );
    }

    #[test]
    fn invalid_env_offset_formats_and_matches() {
        let err = cintxRsError::InvalidEnvOffset {
            slot: "PTR_EXP",
            offset: 128,
            env_len: 64,
        };
        assert!(matches!(
            err,
            cintxRsError::InvalidEnvOffset {
                slot: "PTR_EXP",
                offset: 128,
                env_len: 64
            }
        ));
        assert_eq!(
            err.to_string(),
            "invalid env offset for PTR_EXP: offset=128, env_len=64"
        );
    }

    #[test]
    fn buffer_too_small_formats_and_matches() {
        let err = cintxRsError::BufferTooSmall {
            required: 256,
            provided: 128,
        };
        assert!(matches!(
            err,
            cintxRsError::BufferTooSmall {
                required: 256,
                provided: 128
            }
        ));
        assert_eq!(
            err.to_string(),
            "buffer too small: required=256, provided=128"
        );
    }
}
