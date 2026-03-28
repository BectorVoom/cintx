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

    pub fn memory_limit_bytes(mut self, memory_limit_bytes: Option<usize>) -> Self {
        self.options.memory_limit_bytes = memory_limit_bytes;
        self
    }

    pub fn chunk_size_override(mut self, chunk_size_override: Option<usize>) -> Self {
        self.options.chunk_size_override = chunk_size_override;
        self
    }

    pub fn build(self) -> SessionRequest<'basis> {
        SessionRequest::new(
            self.operator,
            self.representation,
            self.basis,
            self.shells,
            self.options,
        )
    }
}
