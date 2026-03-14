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

use crate::errors::LibcintRsError;

pub type ContractResult<T> = Result<T, LibcintRsError>;

pub fn validate_dims(expected: &[usize], provided: &[usize]) -> ContractResult<()> {
    if expected == provided {
        return Ok(());
    }

    Err(LibcintRsError::DimsBufferMismatch {
        expected: expected.to_vec(),
        provided: provided.to_vec(),
    })
}
