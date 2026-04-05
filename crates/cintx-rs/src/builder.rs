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

    pub fn from_request(request: &SessionRequest<'basis>) -> Self {
        Self {
            operator: request.operator(),
            representation: request.representation(),
            basis: request.basis(),
            shells: request.shells().clone(),
            options: request.options().clone(),
        }
    }

    pub fn options(mut self, options: ExecutionOptions) -> Self {
        self.options = options;
        self
    }

    pub fn profile_label(mut self, profile_label: &'static str) -> Self {
        self.options.profile_label = Some(profile_label);
        self
    }

    pub fn clear_profile_label(mut self) -> Self {
        self.options.profile_label = None;
        self
    }

    pub fn memory_limit(mut self, memory_limit_bytes: usize) -> Self {
        self.options.memory_limit_bytes = Some(memory_limit_bytes);
        self
    }

    pub fn memory_limit_bytes(mut self, memory_limit_bytes: Option<usize>) -> Self {
        self.options.memory_limit_bytes = memory_limit_bytes;
        self
    }

    pub fn clear_memory_limit(mut self) -> Self {
        self.options.memory_limit_bytes = None;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.options.chunk_size_override = Some(chunk_size);
        self
    }

    pub fn chunk_size_override(mut self, chunk_size_override: Option<usize>) -> Self {
        self.options.chunk_size_override = chunk_size_override;
        self
    }

    pub fn clear_chunk_size(mut self) -> Self {
        self.options.chunk_size_override = None;
        self
    }

    /// Set the F12/STG/YP zeta parameter.
    ///
    /// When set, `operator_env_params.f12_zeta` is populated on the `ExecutionPlan`
    /// for F12-family operators. Must be non-zero for F12 calls.
    pub fn f12_zeta(mut self, zeta: f64) -> Self {
        self.options.f12_zeta = Some(zeta);
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

#[cfg(test)]
mod tests {
    use super::SessionBuilder;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let shell_a = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[1.0]), arc_f64(&[1.0, 0.5])).unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[0.8]), arc_f64(&[0.7, 0.3])).unwrap(),
        );

        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice()),
        )
        .unwrap();
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        (basis, shells)
    }

    #[test]
    fn builder_propagates_option_composition_into_request() {
        let (basis, shells) = sample_basis(Representation::Cart);

        let request = SessionBuilder::new(OperatorId::new(0), Representation::Cart, &basis, shells)
            .profile_label("phase-03-safe")
            .memory_limit(4096)
            .chunk_size(8)
            .build();

        assert_eq!(request.options().profile_label, Some("phase-03-safe"));
        assert_eq!(request.options().memory_limit_bytes, Some(4096));
        assert_eq!(request.options().chunk_size_override, Some(8));
        assert_eq!(request.operator(), OperatorId::new(0));
        assert_eq!(request.representation(), Representation::Cart);
        assert_eq!(request.shells().len(), 2);
    }

    #[test]
    fn builder_clear_helpers_remove_optional_overrides() {
        let (basis, shells) = sample_basis(Representation::Spheric);

        let request = SessionBuilder::new(OperatorId::new(1), Representation::Spheric, &basis, shells)
            .profile_label("temporary")
            .memory_limit(1024)
            .chunk_size(2)
            .clear_profile_label()
            .clear_memory_limit()
            .clear_chunk_size()
            .build();

        assert_eq!(request.options().profile_label, None);
        assert_eq!(request.options().memory_limit_bytes, None);
        assert_eq!(request.options().chunk_size_override, None);
    }

    #[test]
    fn builder_from_request_rebuilds_without_mutating_original_contract() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let original = SessionBuilder::new(OperatorId::new(2), Representation::Cart, &basis, shells)
            .profile_label("original")
            .memory_limit(256)
            .chunk_size(4)
            .build();

        let rebuilt = SessionBuilder::from_request(&original)
            .memory_limit(512)
            .build();

        assert_eq!(original.options().memory_limit_bytes, Some(256));
        assert_eq!(rebuilt.options().memory_limit_bytes, Some(512));
        assert_eq!(rebuilt.options().profile_label, Some("original"));
        assert_eq!(rebuilt.options().chunk_size_override, Some(4));
        assert_eq!(rebuilt.operator(), original.operator());
        assert_eq!(rebuilt.representation(), original.representation());
        assert_eq!(rebuilt.shells().len(), original.shells().len());
        assert!(std::ptr::eq(rebuilt.basis(), original.basis()));
    }

    /// Verify that the f12_zeta builder method propagates the zeta value into
    /// ExecutionOptions so the safe API can pass it through to operator_env_params.
    #[test]
    fn builder_f12_zeta_propagates_into_options() {
        let (basis, shells) = sample_basis(Representation::Spheric);

        let request = SessionBuilder::new(OperatorId::new(3), Representation::Spheric, &basis, shells)
            .f12_zeta(1.5)
            .build();

        assert_eq!(
            request.options().f12_zeta,
            Some(1.5),
            "f12_zeta must be carried in ExecutionOptions after builder call"
        );
    }
}
