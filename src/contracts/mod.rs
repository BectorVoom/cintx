pub mod atom;
pub mod basis;
pub mod operator;
pub mod representation;
pub mod shell;

pub use atom::Atom;
pub use basis::BasisSet;
pub use operator::{IntegralFamily, Operator, OperatorKind};
pub use representation::Representation;
pub use shell::{Shell, ShellPrimitive};

pub type ContractResult<T> = Result<T, ContractError>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    #[error("invalid input for {field}: {reason}")]
    InvalidInput { field: &'static str, reason: String },
    #[error("invalid layout for {item}: expected {expected}, got {got}")]
    InvalidLayout {
        item: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{field} index {index} is out of bounds for {collection} length {len}")]
    OutOfBounds {
        field: &'static str,
        index: usize,
        len: usize,
        collection: &'static str,
    },
    #[error("unsupported {field}: {value}")]
    Unsupported {
        field: &'static str,
        value: &'static str,
    },
}
