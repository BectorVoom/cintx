use crate::backend::{self, ResolvedBackend};
use crate::kernels;
use crate::resident_cache::DeviceResidentCache;
use crate::specialization::SpecializationKey;
use crate::transform;
use cintx_core::cintxRsError;
#[cfg(feature = "with-4c1e")]
use cintx_core::Representation;
use cintx_runtime::{
    BackendExecutor, BackendIntent, ExecutionIo, ExecutionPlan, ExecutionStats, OutputOwnership,
    WorkspaceBytes,
};

pub const CUBECL_RUNTIME_PROFILE: &str = "cpu";

/// Lightweight cache that resolves a `ResolvedBackend` from a `BackendIntent`.
///
/// Currently performs live resolution on every call. A future revision may
/// cache the live client handle across calls.
#[derive(Debug, Default)]
pub struct BackendCache;

impl BackendCache {
    pub fn new() -> Self {
        Self
    }

    /// Resolve a backend from the given intent, constructing a live client.
    pub fn resolve(&self, intent: &BackendIntent) -> Result<ResolvedBackend, cintxRsError> {
        ResolvedBackend::from_intent(intent)
    }
}

#[derive(Debug, Default)]
pub struct CubeClExecutor {
    resident_cache: DeviceResidentCache,
    backend_cache: BackendCache,
}

impl CubeClExecutor {
    pub fn new() -> Self {
        Self {
            resident_cache: DeviceResidentCache::new(),
            backend_cache: BackendCache::new(),
        }
    }

    pub fn resident_cache(&self) -> &DeviceResidentCache {
        &self.resident_cache
    }

    /// Resolve the `ResolvedBackend` from the executor's backend cache.
    ///
    /// Reads `CINTX_BACKEND` env var (or defaults to Wgpu) and constructs a
    /// live client handle via `BackendCache::resolve`.
    fn resolve_backend(&self) -> Result<ResolvedBackend, cintxRsError> {
        let backend_kind = backend::resolve_backend_kind();
        let intent = BackendIntent {
            backend: backend_kind,
            selector: "auto".to_owned(),
        };
        self.backend_cache.resolve(&intent)
    }

    /// Check that the backend supports f64 compute (SHADER_F64).
    ///
    /// wgpu path: gates on SHADER_F64 capability.
    /// CPU path: always passes (native f64 support).
    fn check_f64_capability(
        &self,
        backend: &ResolvedBackend,
        _plan: &ExecutionPlan<'_>,
    ) -> Result<(), cintxRsError> {
        match backend {
            ResolvedBackend::Wgpu(_client, _features) => {
                // Gate wgpu dispatch on SHADER_F64 capability. The feature list
                // was captured at bootstrap and stored alongside the client.
                check_shader_f64_in_features(backend.wgpu_features())
            }
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(_client) => Ok(()), // CPU always supports f64 natively.
        }
    }

    #[cfg(feature = "with-4c1e")]
    fn ensure_validated_4c1e(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        if !matches!(
            plan.representation,
            Representation::Cart | Representation::Spheric
        ) {
            return Err(validated_4c1e_error("representation must be cart/sph"));
        }
        if !plan.descriptor.entry.component_rank.trim().is_empty()
            && plan.descriptor.entry.component_rank != "scalar"
        {
            return Err(validated_4c1e_error("component rank must be scalar"));
        }
        // Validated4C1E requires max(l)<=4.
        if plan
            .shells
            .as_slice()
            .iter()
            .any(|shell| shell.ang_momentum > 4)
        {
            return Err(validated_4c1e_error("max(l)>4"));
        }

        Ok(())
    }

    fn ensure_supported_family(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        let canonical_family = plan.descriptor.entry.canonical_family;
        if canonical_family == "4c1e" {
            #[cfg(feature = "with-4c1e")]
            {
                self.ensure_validated_4c1e(plan)?;
            }
            #[cfg(not(feature = "with-4c1e"))]
            return Err(cintxRsError::UnsupportedApi {
                requested: "4c1e requires feature `with-4c1e`".to_owned(),
            });
        }

        if !kernels::supports_canonical_family(canonical_family) {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "CubeCL executor family {canonical_family} is not enabled in the current feature profile"
                ),
            });
        }

        Ok(())
    }
}

#[cfg(feature = "with-4c1e")]
fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("outside Validated4C1E ({reason})"),
    }
}

/// Factored SHADER_F64 capability check for testability.
///
/// Returns `UnsupportedApi` with `"wgpu-capability:missing_shader_f64"` when
/// `SHADER_F64` is absent from the provided feature list. This function is
/// called by `check_f64_capability` for the wgpu arm and exposed for direct
/// unit testing without requiring GPU hardware.
pub fn check_shader_f64_in_features(features: &[String]) -> Result<(), cintxRsError> {
    if !features.iter().any(|f| f == "SHADER_F64") {
        return Err(cintxRsError::UnsupportedApi {
            requested: "wgpu-capability:missing_shader_f64".to_owned(),
        });
    }
    Ok(())
}

impl BackendExecutor for CubeClExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool {
        kernels::supports_canonical_family(plan.descriptor.entry.canonical_family)
            && plan
                .descriptor
                .entry
                .supports_representation(plan.representation)
    }

    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError> {
        let backend = self.resolve_backend()?;
        self.check_f64_capability(&backend, plan)?;
        self.ensure_supported_family(plan)?;
        Ok(WorkspaceBytes(plan.workspace.bytes))
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        let backend = self.resolve_backend()?;
        self.check_f64_capability(&backend, plan)?;
        self.ensure_supported_family(plan)?;
        io.ensure_output_contract()?;

        // D-06: ownership contract enforcement (unchanged from previous executor).
        if io.backend_output_ownership() != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_executor",
                detail: "backend_output must remain staging-only".to_owned(),
            });
        }
        if io.final_write_ownership() != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_executor",
                detail: "CompatFinalWrite must remain owned by compat layout".to_owned(),
            });
        }

        let specialization = SpecializationKey::from_plan(plan);
        let _resident = self.resident_cache.resident_metadata(
            "auto",
            plan.basis,
            plan.representation,
        );

        // EXEC-06: Direct staging pass — no TransferPlan::stage_device_buffers.
        let staging = io.staging_output();
        let mut stats = kernels::launch_family(&backend, plan, &specialization, staging)?;

        // Backend output stays staging-only; compat owns final flat writes.
        // Spinor transforms are applied inside each kernel launcher (Plan 04+) because
        // they require per-shell l and kappa. apply_representation_transform only handles
        // Cart (no-op) and Spheric (c2s). Skip for Spinor to avoid double-transform.
        if !matches!(plan.representation, cintx_core::Representation::Spinor) {
            transform::apply_representation_transform(plan.representation, staging)?;
        }

        stats.peak_workspace_bytes = stats.peak_workspace_bytes.max(io.workspace().len());
        stats.planned_batches = io.chunk().work_unit_count.max(1);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, FallibleBuffer, query_workspace};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation, shell_count: usize) -> BasisSet {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let mut shells = Vec::with_capacity(shell_count);
        for index in 0..shell_count {
            let l = (index % 3 + 1) as u8;
            shells.push(Arc::new(
                Shell::try_new(
                    0,
                    l,
                    1,
                    1,
                    0,
                    rep,
                    arc_f64(&[1.0 - (index as f64 * 0.05)]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            ));
        }

        BasisSet::try_new(atoms, Arc::from(shells.into_boxed_slice())).unwrap()
    }

    fn shell_tuple_for_first_n(basis: &BasisSet, count: usize) -> ShellTuple {
        ShellTuple::try_from_iter(
            basis
                .shells()
                .iter()
                .take(count)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn build_plan(
        basis: &'static BasisSet,
        operator_id: u32,
        rep: Representation,
        arity: usize,
    ) -> ExecutionPlan<'static> {
        let shells = shell_tuple_for_first_n(basis, arity);
        let query = query_workspace(
            OperatorId::new(operator_id),
            rep,
            basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        ExecutionPlan::new(OperatorId::new(operator_id), rep, basis, shells, query).unwrap()
    }

    #[test]
    fn shader_f64_absent_returns_unsupported_api() {
        // Test that check_shader_f64_in_features returns UnsupportedApi when
        // SHADER_F64 is absent from the feature list.
        //
        // This function is factored out of check_f64_capability so that the
        // SHADER_F64 gate is testable without requiring GPU hardware.
        let features_without_f64: Vec<String> = vec![
            "TIMESTAMP_QUERY".to_owned(),
            "PUSH_CONSTANTS".to_owned(),
        ];
        let result = check_shader_f64_in_features(&features_without_f64);
        assert!(result.is_err());
        match result.unwrap_err() {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("missing_shader_f64"),
                    "Expected 'missing_shader_f64' in error, got: {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got: {other:?}"),
        }

        // Also verify that a feature list WITH SHADER_F64 passes:
        let features_with_f64: Vec<String> = vec![
            "SHADER_F64".to_owned(),
            "TIMESTAMP_QUERY".to_owned(),
        ];
        let result = check_shader_f64_in_features(&features_with_f64);
        assert!(result.is_ok(), "SHADER_F64 present should pass check");

        // Empty feature list should also fail:
        let empty_features: Vec<String> = vec![];
        let result = check_shader_f64_in_features(&empty_features);
        assert!(
            result.is_err(),
            "Empty feature list should fail SHADER_F64 check"
        );
    }

    #[test]
    fn supports_full_phase2_base_families() {
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let executor = CubeClExecutor::new();

        let one_e = build_plan(basis, 0, Representation::Cart, 2);
        let two_e = build_plan(basis, 9, Representation::Cart, 4);
        let two_c2e = build_plan(basis, 12, Representation::Cart, 2);
        let three_c1e = build_plan(basis, 15, Representation::Cart, 3);
        let three_c2e = build_plan(basis, 17, Representation::Cart, 3);

        assert!(executor.supports(&one_e));
        assert!(executor.supports(&two_e));
        assert!(executor.supports(&two_c2e));
        assert!(executor.supports(&three_c1e));
        assert!(executor.supports(&three_c2e));
    }

    #[cfg(not(feature = "with-4c1e"))]
    #[test]
    fn unsupported_4c1e_is_rejected_without_feature() {
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let executor = CubeClExecutor::new();
        let op_4c1e = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("4c1e descriptor should exist")
            .id
            .raw();
        let plan = build_plan(basis, op_4c1e, Representation::Cart, 4);
        // resolve_backend() will fail on wgpu (no GPU), which returns UnsupportedApi.
        // On CPU (CINTX_BACKEND=cpu), it will proceed to the 4c1e family check.
        let err = executor.query_workspace(&plan).unwrap_err();
        assert!(matches!(err, cintxRsError::UnsupportedApi { .. }));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn validated_4c1e_is_supported_with_feature() {
        // Requires CINTX_BACKEND=cpu to avoid wgpu init failure on no-GPU CI.
        if std::env::var("CINTX_BACKEND").as_deref() != Ok("cpu") {
            return; // Skip on non-cpu environments.
        }
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 4)));
        let executor = CubeClExecutor::new();
        let op_4c1e = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("4c1e descriptor should exist")
            .id
            .raw();
        let plan = build_plan(basis, op_4c1e, Representation::Cart, 4);
        assert!(executor.supports(&plan));
        assert!(executor.query_workspace(&plan).is_ok());
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn outside_validated_4c1e_envelope_is_rejected() {
        // Requires CINTX_BACKEND=cpu to avoid wgpu init failure on no-GPU CI.
        if std::env::var("CINTX_BACKEND").as_deref() != Ok("cpu") {
            return; // Skip on non-cpu environments.
        }
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let shells = Arc::from(
            vec![
                Arc::new(
                    Shell::try_new(
                        0,
                        5,
                        1,
                        1,
                        0,
                        Representation::Cart,
                        arc_f64(&[1.0]),
                        arc_f64(&[1.0]),
                    )
                    .unwrap(),
                ),
                Arc::new(
                    Shell::try_new(
                        0,
                        1,
                        1,
                        1,
                        0,
                        Representation::Cart,
                        arc_f64(&[1.0]),
                        arc_f64(&[1.0]),
                    )
                    .unwrap(),
                ),
                Arc::new(
                    Shell::try_new(
                        0,
                        1,
                        1,
                        1,
                        0,
                        Representation::Cart,
                        arc_f64(&[1.0]),
                        arc_f64(&[1.0]),
                    )
                    .unwrap(),
                ),
                Arc::new(
                    Shell::try_new(
                        0,
                        1,
                        1,
                        1,
                        0,
                        Representation::Cart,
                        arc_f64(&[1.0]),
                        arc_f64(&[1.0]),
                    )
                    .unwrap(),
                ),
            ]
            .into_boxed_slice(),
        );
        let basis = BasisSet::try_new(atoms, shells).unwrap();
        let basis = Box::leak(Box::new(basis));

        let executor = CubeClExecutor::new();
        let op_4c1e = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("4c1e descriptor should exist")
            .id
            .raw();
        let plan = build_plan(basis, op_4c1e, Representation::Cart, 4);
        let err = executor.query_workspace(&plan).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") || requested.contains("wgpu")
        ));
    }

    #[test]
    fn representation_transforms_keep_staging_only_contract() {
        // This test requires CINTX_BACKEND=cpu since the execute path now calls
        // resolve_backend() which will fail on wgpu in no-GPU environments.
        if std::env::var("CINTX_BACKEND").as_deref() != Ok("cpu") {
            return; // Skip on non-cpu environments.
        }
        let executor = CubeClExecutor::new();

        // Cart path: identity transform over deterministic cart staging seed.
        let cart_basis = Box::leak(Box::new(sample_basis(Representation::Cart, 2)));
        let cart_plan = build_plan(cart_basis, 0, Representation::Cart, 2);
        let cart_chunk = cart_plan.workspace.chunks[0].clone();
        let mut cart_staging = vec![0.0; 8];
        let mut cart_workspace = FallibleBuffer::try_uninit(
            cart_plan.workspace.bytes.max(1),
            cart_plan.workspace.alignment,
        )
        .unwrap();
        let mut cart_io = ExecutionIo::new(
            &cart_chunk,
            &mut cart_staging,
            &mut cart_workspace,
            cart_plan.dispatch,
        )
        .unwrap();
        executor.execute(&cart_plan, &mut cart_io).unwrap();
        assert_eq!(
            cart_io.backend_output_ownership(),
            OutputOwnership::BackendStagingOnly
        );
        assert_eq!(
            cart_io.final_write_ownership(),
            OutputOwnership::CompatFinalWrite
        );

        // Spheric path: c2s transform mutates staging values.
        let sph_basis = Box::leak(Box::new(sample_basis(Representation::Spheric, 2)));
        let sph_plan = build_plan(sph_basis, 1, Representation::Spheric, 2);
        let sph_chunk = sph_plan.workspace.chunks[0].clone();
        let mut sph_staging = vec![0.0; 8];
        let mut sph_workspace = FallibleBuffer::try_uninit(
            sph_plan.workspace.bytes.max(1),
            sph_plan.workspace.alignment,
        )
        .unwrap();
        let mut sph_io = ExecutionIo::new(
            &sph_chunk,
            &mut sph_staging,
            &mut sph_workspace,
            sph_plan.dispatch,
        )
        .unwrap();
        executor.execute(&sph_plan, &mut sph_io).unwrap();

        // Spinor path: interleaved doubles keep real/imag pair semantics.
        let spinor_basis = Box::leak(Box::new(sample_basis(Representation::Spinor, 2)));
        let spinor_plan = build_plan(spinor_basis, 2, Representation::Spinor, 2);
        let spinor_chunk = spinor_plan.workspace.chunks[0].clone();
        let mut spinor_staging = vec![0.0; 8];
        let mut spinor_workspace = FallibleBuffer::try_uninit(
            spinor_plan.workspace.bytes.max(1),
            spinor_plan.workspace.alignment,
        )
        .unwrap();
        let mut spinor_io = ExecutionIo::new(
            &spinor_chunk,
            &mut spinor_staging,
            &mut spinor_workspace,
            spinor_plan.dispatch,
        )
        .unwrap();
        executor.execute(&spinor_plan, &mut spinor_io).unwrap();
        for pair in spinor_staging.chunks_exact(2) {
            assert!((pair[0] + pair[1]).abs() < f64::EPSILON);
        }
    }
}
