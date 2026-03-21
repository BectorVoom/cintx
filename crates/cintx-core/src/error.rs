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

pub type CoreResult<T> = Result<T, CoreError>;
