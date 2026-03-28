//! Builder types for safe request/session scaffolding.

use crate::api::SessionRequest;
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple};
use cintx_runtime::ExecutionOptions;

#[derive(Clone, Debug)]
pub struct SessionBuilder<'basis> {
    operator: OperatorId,
    representation: Representation,
    basis: &'basis BasisSet,
    shells: ShellTuple,
    options: ExecutionOptions,
}

impl<'basis> SessionBuilder<'basis> {
    pub fn new(
        operator: OperatorId,
        representation: Representation,
        basis: &'basis BasisSet,
        shells: ShellTuple,
    ) -> Self {
        Self {
            operator,
            representation,
            basis,
            shells,
            options: ExecutionOptions::default(),
        }
    }

    pub fn options(mut self, options: ExecutionOptions) -> Self {
        self.options = options;
        self
    }

    pub fn build(self) -> SessionRequest<'basis> {
        SessionRequest {
            operator: self.operator,
            representation: self.representation,
            basis: self.basis,
            shells: self.shells,
            options: self.options,
        }
    }
}
