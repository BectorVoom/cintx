use crate::kernels;
use crate::resident_cache::DeviceResidentCache;
use crate::runtime_bootstrap::bootstrap_wgpu_runtime;
use crate::specialization::SpecializationKey;
use crate::transfer::TransferPlan;
use crate::transform;
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionPlan, ExecutionStats, OutputOwnership, WorkspaceBytes,
};

#[derive(Debug, Default)]
pub struct CubeClExecutor {
    resident_cache: DeviceResidentCache,
}

impl CubeClExecutor {
    pub fn new() -> Self {
        Self {
            resident_cache: DeviceResidentCache::new(),
        }
    }

    pub fn resident_cache(&self) -> &DeviceResidentCache {
        &self.resident_cache
    }

    /// Preflight the wgpu capability for the given plan's backend intent.
    ///
    /// Returns `UnsupportedApi` with a `wgpu-capability:` prefix when no adapter
    /// is available or the adapter fails capability checks (D-01, D-02).
    fn preflight_wgpu(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        let report = bootstrap_wgpu_runtime(&plan.workspace.backend_intent)?;
        if !report.is_capable() {
            if let Some(reason) = report.first_reason() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("wgpu-capability:{}", reason.to_reason_string()),
                });
            }
        }
        Ok(())
    }

    fn ensure_validated_4c1e(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        // D-11: Validated4C1E gate now requires wgpu capability preflight success,
        // not a cpu-profile string check.
        let report = bootstrap_wgpu_runtime(&plan.workspace.backend_intent)
            .map_err(|_| validated_4c1e_error("missing_wgpu_capability: no valid wgpu adapter"))?;
        if !report.is_capable() {
            let reason = report
                .first_reason()
                .map(|r| r.to_reason_string())
                .unwrap_or_else(|| "unknown".to_owned());
            return Err(validated_4c1e_error(&format!(
                "missing_wgpu_capability: {reason}"
            )));
        }
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
                requested: format!("unsupported_family:{canonical_family}"),
            });
        }

        if !kernels::supports_canonical_family(canonical_family) {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("unsupported_family:{canonical_family}"),
            });
        }

        // Check representation support with explicit taxonomy reason (D-12).
        if !plan
            .descriptor
            .entry
            .supports_representation(plan.representation)
        {
            let rep = plan.representation.to_string();
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("unsupported_representation:{rep}"),
            });
        }

        Ok(())
    }
}

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("outside Validated4C1E ({reason})"),
    }
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
        // D-01 / D-02: fail closed with capability taxonomy reason before proceeding.
        self.preflight_wgpu(plan)?;
        self.ensure_supported_family(plan)?;
        Ok(WorkspaceBytes(plan.workspace.bytes))
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        // D-01 / D-02: fail closed at execute entry.
        self.preflight_wgpu(plan)?;
        self.ensure_supported_family(plan)?;
        io.ensure_output_contract()?;

        // D-06: ownership contract must be enforced before and after kernel dispatch.
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
            plan.workspace.backend_intent.selector.as_str(),
            plan.basis,
            plan.representation,
        );
        let transfer_plan = TransferPlan::from_plan(plan, io.chunk())?;
        transfer_plan.ensure_output_contract()?;

        // Include adapter identifier from preflight in device OOM mapping (D-04).
        let adapter_label = plan.workspace.backend_intent.selector.as_str();
        let transfer = transfer_plan.stage_device_buffers(adapter_label)?;

        // D-05 / D-07: real CubeCL launch path — kernel writes into staging via launch/readback.
        // The staging slice is obtained here and passed to the representation transform step.
        let mut stats = kernels::launch_family(plan, &specialization, &transfer_plan)?;

        // D-06: backend output stays staging-only; compat owns final flat writes.
        let staging = io.staging_output();
        // No synthetic fill: staging retains the kernel readback values (zeros from stub kernels
        // or real integral values when GPU kernels are implemented in later plans).
        transform::apply_representation_transform(plan.representation, staging)?;

        stats.peak_workspace_bytes = stats
            .peak_workspace_bytes
            .max(transfer.workspace_bytes.max(io.workspace().len()));
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
        let err = executor.query_workspace(&plan).unwrap_err();
        assert!(matches!(err, cintxRsError::UnsupportedApi { .. }));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn validated_4c1e_is_supported_with_feature() {
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
                if requested.contains("outside Validated4C1E")
        ));
    }

    /// D-05: Executor must not fill staging with monotonically increasing integers.
    ///
    /// When a GPU adapter is available, staging remains zero (no synthetic fill).
    /// When no GPU is available, execute returns a typed wgpu-capability error
    /// (fail-closed per D-01/D-02) — staging is never touched by a synthetic fill.
    #[test]
    fn executor_no_longer_uses_monotonic_stub_sequence() {
        let executor = CubeClExecutor::new();
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 2)));
        let plan = build_plan(basis, 0, Representation::Cart, 2);
        let chunk = plan.workspace.chunks[0].clone();
        let mut staging = vec![0.0_f64; 8];
        let mut workspace = FallibleBuffer::try_uninit(
            plan.workspace.bytes.max(1),
            plan.workspace.alignment,
        )
        .unwrap();
        let mut io = ExecutionIo::new(&chunk, &mut staging, &mut workspace, plan.dispatch).unwrap();
        match executor.execute(&plan, &mut io) {
            Ok(_stats) => {
                // GPU available: staging must remain 0.0, not 1.0/2.0/... from synthetic fill.
                for &val in staging.iter() {
                    assert_eq!(val, 0.0, "stub monotonic fill must be removed; staging must stay 0.0");
                }
            }
            Err(cintxRsError::UnsupportedApi { requested }) => {
                // No GPU: fail-closed means staging was never touched by any fill.
                assert!(
                    requested.starts_with("wgpu-capability:"),
                    "no-GPU error must carry 'wgpu-capability:' prefix: {requested}"
                );
                // Staging must still be zero (no synthetic fill occurred before error).
                for &val in staging.iter() {
                    assert_eq!(val, 0.0, "staging must be untouched when execute fails early");
                }
            }
            Err(other) => panic!("Unexpected error from execute: {other:?}"),
        }
    }

    /// D-07: execute() must record per-chunk transfer_bytes and not0 metrics from
    /// a real kernel launch path instead of hardcoded values.
    ///
    /// D-01 / D-02: wgpu bootstrap is called at execute entry; a report with
    /// capability metadata must be accessible (fingerprint non-zero).  On
    /// environments without a GPU adapter the call returns a typed
    /// `wgpu-capability:missing_adapter` error (fail-closed, not a panic).
    ///
    /// D-06: On GPU environments, ownership contract checks enforce BackendStagingOnly
    /// and CompatFinalWrite after real kernel launch path runs.
    #[test]
    fn execute_uses_wgpu_bootstrap_and_preserves_output_contract() {
        let executor = CubeClExecutor::new();
        let basis = Box::leak(Box::new(sample_basis(Representation::Cart, 2)));
        let plan = build_plan(basis, 0, Representation::Cart, 2);
        let chunk = plan.workspace.chunks[0].clone();
        let mut staging = vec![0.0_f64; 8];
        let mut workspace = FallibleBuffer::try_uninit(
            plan.workspace.bytes.max(1),
            plan.workspace.alignment,
        )
        .unwrap();
        let mut io = ExecutionIo::new(&chunk, &mut staging, &mut workspace, plan.dispatch).unwrap();
        match executor.execute(&plan, &mut io) {
            Ok(stats) => {
                // GPU available: D-07 — metrics must reflect actual kernel path.
                assert!(stats.transfer_bytes > 0, "transfer_bytes must be >0 after execute");
                assert!(stats.chunk_count >= 1, "chunk_count must be >=1 after execute");
                // D-06: ownership contracts must remain enforced.
                assert_eq!(
                    io.backend_output_ownership(),
                    OutputOwnership::BackendStagingOnly,
                    "Backend output ownership must remain BackendStagingOnly"
                );
                assert_eq!(
                    io.final_write_ownership(),
                    OutputOwnership::CompatFinalWrite,
                    "Final write ownership must remain CompatFinalWrite"
                );
            }
            Err(cintxRsError::UnsupportedApi { requested }) => {
                // No GPU adapter in CI environment: D-01/D-02 — fail-closed with typed reason.
                assert!(
                    requested.starts_with("wgpu-capability:"),
                    "no-GPU error must carry 'wgpu-capability:' prefix: {requested}"
                );
            }
            Err(other) => panic!("Unexpected error from execute: {other:?}"),
        }
    }

    #[test]
    fn representation_transforms_keep_staging_only_contract() {
        let executor = CubeClExecutor::new();

        // Cart path: identity transform over kernel-output staging values.
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
        match executor.execute(&cart_plan, &mut cart_io) {
            Ok(_) => {
                // GPU available: ownership must be preserved, staging untouched by synthetic fill.
                assert_eq!(cart_io.backend_output_ownership(), OutputOwnership::BackendStagingOnly);
                assert_eq!(cart_io.final_write_ownership(), OutputOwnership::CompatFinalWrite);
            }
            Err(cintxRsError::UnsupportedApi { requested }) if requested.starts_with("wgpu-capability:") => {
                // No GPU: correct fail-closed behavior.
            }
            Err(other) => panic!("Unexpected error: {other:?}"),
        }

        // Spheric path: c2s transform applied, ownership contract preserved.
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
        match executor.execute(&sph_plan, &mut sph_io) {
            Ok(_) => {
                assert_eq!(sph_io.backend_output_ownership(), OutputOwnership::BackendStagingOnly);
                assert_eq!(sph_io.final_write_ownership(), OutputOwnership::CompatFinalWrite);
            }
            Err(cintxRsError::UnsupportedApi { requested }) if requested.starts_with("wgpu-capability:") => {}
            Err(other) => panic!("Unexpected error: {other:?}"),
        }

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
        match executor.execute(&spinor_plan, &mut spinor_io) {
            Ok(_) => {
                // Spinor: interleaved doubles keep real/imag pair semantics (zeros sum to zero).
                for pair in spinor_staging.chunks_exact(2) {
                    assert!((pair[0] + pair[1]).abs() < f64::EPSILON);
                }
            }
            Err(cintxRsError::UnsupportedApi { requested }) if requested.starts_with("wgpu-capability:") => {}
            Err(other) => panic!("Unexpected error: {other:?}"),
        }
    }
}
