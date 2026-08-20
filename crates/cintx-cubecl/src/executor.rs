use crate::backend::ResolvedBackend;
use crate::batch_pilot::{
    EriSsssInput, OverlapSsInput, PilotOutputArena, PilotOutputArenaStats, SsBatchChunkOutput,
    run_eri_ssss_batch_chunks, run_overlap_ss_batch, run_ss_batch_chunks_with_output_arena,
};
use crate::kernels;
use crate::resident_cache::DeviceResidentCache;
use crate::specialization::SpecializationKey;
use crate::transform;
#[cfg(feature = "with-4c1e")]
use cintx_core::Representation;
use cintx_core::{PrecisionKind, cintxRsError};
use cintx_runtime::{
    BackendExecutor, BackendIntent, ExecutionIo, ExecutionPlan, ExecutionStats, OutputOwnership,
    WorkspaceBytes,
};
use std::sync::{Arc, Mutex, RwLock};

pub const CUBECL_RUNTIME_PROFILE: &str = "cpu";

/// Long-lived CubeCL clients keyed by the query-time backend intent.
///
/// Backend bootstrap is a cold-path operation. Retaining the client here keeps
/// query and evaluation on the same device and removes that work from warm
/// submissions.
#[derive(Default)]
pub struct BackendCache {
    entries: RwLock<Vec<(BackendIntent, Arc<ResolvedBackend>)>>,
}

impl std::fmt::Debug for BackendCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries = self
            .entries
            .read()
            .map(|entries| entries.len())
            .unwrap_or(0);
        formatter
            .debug_struct("BackendCache")
            .field("entries", &entries)
            .finish()
    }
}

impl BackendCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a backend from the supplied intent, bootstrapping it once.
    pub fn resolve(&self, intent: &BackendIntent) -> Result<Arc<ResolvedBackend>, cintxRsError> {
        if let Some(existing) = self
            .entries
            .read()
            .expect("backend cache poisoned")
            .iter()
            .find_map(|(cached_intent, backend)| {
                (cached_intent == intent).then(|| Arc::clone(backend))
            })
        {
            return Ok(existing);
        }

        let resolved = Arc::new(ResolvedBackend::from_intent(intent)?);
        let mut entries = self.entries.write().expect("backend cache poisoned");
        if let Some((_, existing)) = entries
            .iter()
            .find(|(cached_intent, _)| cached_intent == intent)
        {
            return Ok(Arc::clone(existing));
        }
        entries.push((intent.clone(), Arc::clone(&resolved)));
        Ok(resolved)
    }

    pub fn len(&self) -> usize {
        self.entries.read().expect("backend cache poisoned").len()
    }
}

#[derive(Debug, Default)]
pub struct CubeClExecutor {
    resident_cache: DeviceResidentCache,
    backend_cache: BackendCache,
    /// Output-only staging for the narrow Cartesian s-s batch pilot. The
    /// mutex deliberately spans submission through its collective readback so
    /// one reusable output handle cannot be rebound by a concurrent batch.
    pilot_output_arena: Mutex<PilotOutputArena>,
}

impl CubeClExecutor {
    pub fn new() -> Self {
        Self {
            resident_cache: DeviceResidentCache::new(),
            backend_cache: BackendCache::new(),
            pilot_output_arena: Mutex::new(PilotOutputArena::default()),
        }
    }

    pub fn resident_cache(&self) -> &DeviceResidentCache {
        &self.resident_cache
    }

    /// Number of backend clients retained for distinct query-time intents.
    pub fn backend_cache_entries(&self) -> usize {
        self.backend_cache.len()
    }

    /// Aggregate output-staging reuse retained by the Cartesian s-s batch pilot.
    /// Descriptor uploads are intentionally excluded because they remain
    /// request-specific and are never structurally cached.
    pub fn pilot_output_arena_stats(&self) -> PilotOutputArenaStats {
        self.pilot_output_arena
            .lock()
            .expect("pilot output arena poisoned")
            .stats()
    }

    /// Run the verified s-s overlap batch pilot through this executor's cached client.
    ///
    /// This is intentionally narrow while general shell-pair descriptors and transposed
    /// recurrence scratch are being migrated.
    pub fn execute_overlap_ss_batch(
        &self,
        intent: &BackendIntent,
        inputs: &[OverlapSsInput],
    ) -> Result<Vec<f64>, cintxRsError> {
        let backend = self.backend_cache.resolve(intent)?;
        self.check_f64_capability(&backend)?;
        run_overlap_ss_batch(&backend, inputs)
    }

    /// Submit all planned overlap chunks before the pilot's single collective readback.
    pub fn execute_overlap_ss_batch_chunks(
        &self,
        intent: &BackendIntent,
        chunks: &[&[OverlapSsInput]],
    ) -> Result<SsBatchChunkOutput, cintxRsError> {
        let backend = self.backend_cache.resolve(intent)?;
        self.check_f64_capability(&backend)?;
        self.execute_ss_batch_chunks_with_output_arena(&backend, intent, chunks, false)
    }

    /// Submit all planned single-contraction Cartesian kinetic s-s chunks before one readback.
    pub fn execute_kinetic_ss_batch_chunks(
        &self,
        intent: &BackendIntent,
        chunks: &[&[OverlapSsInput]],
    ) -> Result<SsBatchChunkOutput, cintxRsError> {
        let backend = self.backend_cache.resolve(intent)?;
        self.check_f64_capability(&backend)?;
        self.execute_ss_batch_chunks_with_output_arena(&backend, intent, chunks, true)
    }

    /// Select the verified Cartesian s-s pilot specialization for a homogeneous bucket.
    pub fn execute_ss_batch_chunks(
        &self,
        intent: &BackendIntent,
        chunks: &[&[OverlapSsInput]],
        kinetic: bool,
    ) -> Result<SsBatchChunkOutput, cintxRsError> {
        let backend = self.backend_cache.resolve(intent)?;
        self.check_f64_capability(&backend)?;
        self.execute_ss_batch_chunks_with_output_arena(&backend, intent, chunks, kinetic)
    }

    fn execute_ss_batch_chunks_with_output_arena(
        &self,
        backend: &ResolvedBackend,
        intent: &BackendIntent,
        chunks: &[&[OverlapSsInput]],
        kinetic: bool,
    ) -> Result<SsBatchChunkOutput, cintxRsError> {
        let mut arena = self
            .pilot_output_arena
            .lock()
            .expect("pilot output arena poisoned");
        Ok(run_ss_batch_chunks_with_output_arena(
            backend, chunks, kinetic, intent, &mut arena,
        ))
    }

    /// Submit all primitive Cartesian `(s s | s s)` chunks before one
    /// collective readback.  This is intentionally separate from the general
    /// two-electron dispatcher until all 2e descriptor/output variants carry
    /// their own parity gates.
    pub fn execute_eri_ssss_batch_chunks(
        &self,
        intent: &BackendIntent,
        chunks: &[&[EriSsssInput]],
    ) -> Result<SsBatchChunkOutput, cintxRsError> {
        let backend = self.backend_cache.resolve(intent)?;
        self.check_f64_capability(&backend)?;
        Ok(run_eri_ssss_batch_chunks(&backend, chunks))
    }

    /// Resolve the client chosen at query time, never from a later env read.
    fn resolve_backend(
        &self,
        plan: &ExecutionPlan<'_>,
    ) -> Result<Arc<ResolvedBackend>, cintxRsError> {
        self.backend_cache.resolve(&plan.workspace.backend_intent)
    }

    /// Check that the backend supports f64 compute (SHADER_F64).
    ///
    /// wgpu/metal path: gates on SHADER_F64 capability.
    /// CPU path: always passes (native f64 support).
    /// CUDA path: f64 capable; runtime accept-with-failure.
    /// ROCm path: dev-host runtime-verified; accept-with-failure.
    fn check_f64_capability(&self, backend: &ResolvedBackend) -> Result<(), cintxRsError> {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(_) => Ok(()),
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(_, _) => {
                // Gate wgpu dispatch on SHADER_F64 capability. The feature list
                // was captured at bootstrap and stored alongside the client.
                check_shader_f64_in_features(backend.wgpu_features())
            }
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(_) => Ok(()), // f64 capable; runtime accept-with-failure.
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(_) => Ok(()), // dev-host runtime-verified; accept-with-failure.
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
        }
    }

    /// Precision-aware capability check (D-10 / PREC-06).
    ///
    /// For `PrecisionKind::F32`: returns `Ok(())` immediately — f32 is WebGPU-baseline
    /// universal, so no `SHADER_F64` gate is needed. This unlocks wgpu adapters that
    /// lack native f64 capability for the f32 evaluation path.
    ///
    /// For `PrecisionKind::F64` (default): delegates byte-identically to
    /// `check_f64_capability`, which enforces the `SHADER_F64` requirement on wgpu/metal
    /// (Pitfall 3 / PREC-04: the f64 arm stays exactly as before this change).
    ///
    /// The two call sites in `query_workspace` and `execute` both call this method;
    /// `check_f64_capability` is kept intact and private (not removed) to remain
    /// independently testable.
    fn check_capability(
        &self,
        backend: &ResolvedBackend,
        plan: &ExecutionPlan<'_>,
    ) -> Result<(), cintxRsError> {
        if plan.precision == PrecisionKind::F32 {
            // f32 is WebGPU-baseline universal — no SHADER_F64 gate required (D-10).
            return Ok(());
        }
        // F64 arm: delegate to the byte-identical existing capability check.
        self.check_f64_capability(backend)
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
        // Validated 4c1e angular-momentum ceiling (Phase 25 FND-02 / D-02).
        //
        // Before FND-02 the host Rys engine panicked for nroots>5, so the gate
        // capped at max(l)<=4 (g). With the host Wheeler nroots 6..12 engine
        // landed (rys_wheeler.rs) and validated on the dedicated nroots sweep, the
        // gate is raised to the max angular momentum those roots support — D-02's
        // forward-looking foundation (validated on the nroots sweep, NOT a g/h family
        // parity test, per D-03). For a 4c1e quartet of homogeneous angular momentum
        // l, the Rys order is nroots = (4l)/2 + 1 = 2l+1; the validated ceiling
        // nroots<=12 admits up to l=5 (h: 2*5+1=11). l=6 (i) would need nroots=13,
        // which routes to the quadmath path the vendor build does not compile — so
        // the gate stays BOUNDED at the validated ceiling (NOT unbounded above it).
        if plan
            .shells
            .as_slice()
            .iter()
            .any(|shell| shell.ang_momentum > VALIDATED_4C1E_MAX_L)
        {
            return Err(validated_4c1e_error("max(l)>5"));
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

/// Max angular momentum admitted by the Validated4C1E gate (Phase 25 FND-02 / D-02).
///
/// l=5 (h): a homogeneous 4c1e quartet needs Rys order nroots = 2l+1 = 11 <= 12, the
/// vendor-validated ceiling (quadmath disabled => 12). l=6 (i) would need nroots=13
/// (uncompiled quadmath path), so the gate is bounded here, not unbounded.
#[cfg(feature = "with-4c1e")]
const VALIDATED_4C1E_MAX_L: u8 = 5;

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
        let backend = self.resolve_backend(plan)?;
        self.check_capability(&backend, plan)?;
        self.ensure_supported_family(plan)?;
        Ok(WorkspaceBytes(plan.workspace.bytes))
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        let backend = self.resolve_backend(plan)?;
        self.check_capability(&backend, plan)?;
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
        let _resident =
            self.resident_cache
                .resident_metadata("auto", plan.basis, plan.representation);

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
    use cintx_runtime::{
        BackendIntent, BackendKind, ExecutionOptions, FallibleBuffer, query_workspace,
    };
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn backend_cache_reuses_query_intent_client() {
        let cache = BackendCache::new();
        let intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "auto".to_owned(),
        };

        let first = cache.resolve(&intent).expect("cpu backend resolves");
        let second = cache.resolve(&intent).expect("cached cpu backend resolves");

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn overlap_ss_batch_pilot_uses_the_executor_client_cache() {
        let executor = CubeClExecutor::new();
        let intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "auto".to_owned(),
        };
        let inputs = vec![
            OverlapSsInput {
                exponents_i: Arc::from(vec![0.5].into_boxed_slice()),
                exponents_j: Arc::from(vec![0.7].into_boxed_slice()),
                coefficients_i: Arc::from(vec![1.0].into_boxed_slice()),
                coefficients_j: Arc::from(vec![0.8].into_boxed_slice()),
                center_i: [0.0, 0.0, 0.0],
                center_j: [0.0, 0.0, 1.4],
            };
            65
        ];

        let output = executor
            .execute_overlap_ss_batch_chunks(&intent, &[&inputs])
            .unwrap();

        assert_eq!(output.chunks[0].len(), inputs.len());
        assert!(
            output.chunks[0]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert_eq!(output.output_staging_allocations, 1);

        let warm = executor
            .execute_overlap_ss_batch_chunks(&intent, &[&inputs[..1]])
            .unwrap();
        assert_eq!(warm.output_staging_allocations, 0);
        assert_eq!(warm.output_staging_reuses, 1);

        let mut larger = inputs.clone();
        larger.push(inputs[0].clone());
        let grown = executor
            .execute_overlap_ss_batch_chunks(&intent, &[&larger])
            .unwrap();
        assert_eq!(grown.output_staging_allocations, 1);
        assert_eq!(grown.output_staging_growths, 1);
        let arena = executor.pilot_output_arena_stats();
        assert_eq!(arena.allocations, 2);
        assert_eq!(arena.reuses, 1);
        assert_eq!(arena.growths, 1);
        assert_eq!(
            arena.retained_bytes,
            larger.len() * std::mem::size_of::<f64>()
        );
        assert_eq!(executor.backend_cache_entries(), 1);
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

    // ─────────────────────────────────────────────────────────────────────────
    // T06-1a: check_capability with F32 precision returns Ok even when the
    // wgpu feature list lacks "SHADER_F64".
    // RED: compile fails until check_capability is added to CubeClExecutor.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn check_capability_f32_bypasses_shader_f64_gate() {
        use crate::backend::cpu_backend::resolve_cpu_client;
        use cintx_core::PrecisionKind;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, query_workspace};

        // Build a minimal plan with precision == F32.
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = std::sync::Arc::from(vec![atom].into_boxed_slice());
        let shell_a = std::sync::Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let shell_b = std::sync::Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells =
            std::sync::Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = Box::leak(Box::new(BasisSet::try_new(atoms, all_shells).unwrap()));
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        let mut plan = cintx_runtime::ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells,
            query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F32;

        let executor = CubeClExecutor::new();

        // Test check_capability directly: F32 must return Ok on the cpu backend.
        // (On wgpu/metal without SHADER_F64, it would also return Ok due to early bypass.)
        #[cfg(feature = "cpu")]
        {
            let cpu_client = resolve_cpu_client().unwrap();
            let backend = ResolvedBackend::Cpu(cpu_client);
            // RED: This call fails to compile until check_capability is defined on CubeClExecutor.
            let result = executor.check_capability(&backend, &plan);
            assert!(
                result.is_ok(),
                "F32 check_capability must return Ok (bypass): {:?}",
                result
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T06-1b: check_capability with F64 precision delegates to check_f64_capability,
    // enforcing SHADER_F64 on wgpu/metal.
    // RED: compile fails until check_capability is added to CubeClExecutor.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn check_capability_f64_delegates_to_check_f64_capability() {
        use crate::backend::cpu_backend::resolve_cpu_client;
        use cintx_core::PrecisionKind;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, query_workspace};

        // Build a minimal plan with precision == F64 (default).
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = std::sync::Arc::from(vec![atom].into_boxed_slice());
        let shell_a = std::sync::Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let shell_b = std::sync::Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
                std::sync::Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells =
            std::sync::Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = Box::leak(Box::new(BasisSet::try_new(atoms, all_shells).unwrap()));
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let query = Box::leak(Box::new(query));
        let mut plan = cintx_runtime::ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            basis,
            shells,
            query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F64; // explicit F64

        let executor = CubeClExecutor::new();

        // Test check_capability directly on CPU: F64+CPU must return Ok
        // (cpu always passes check_f64_capability — byte-identical behavior preserved).
        #[cfg(feature = "cpu")]
        {
            let cpu_client = resolve_cpu_client().unwrap();
            let backend = ResolvedBackend::Cpu(cpu_client);
            // RED: This call fails to compile until check_capability is defined on CubeClExecutor.
            let result = executor.check_capability(&backend, &plan);
            assert!(
                result.is_ok(),
                "F64 check_capability on CPU must return Ok: {:?}",
                result
            );
        }

        // Verify check_shader_f64_in_features still fails closed (frozen behavior).
        let features_without_f64: Vec<String> =
            vec!["TIMESTAMP_QUERY".to_owned(), "PUSH_CONSTANTS".to_owned()];
        let result = check_shader_f64_in_features(&features_without_f64);
        assert!(
            result.is_err(),
            "F64 SHADER_F64 gate must still fail closed"
        );
        match result.unwrap_err() {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("missing_shader_f64"),
                    "F64 path error must mention missing_shader_f64, got: {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got: {other:?}"),
        }
    }

    #[test]
    fn shader_f64_absent_returns_unsupported_api() {
        // Test that check_shader_f64_in_features returns UnsupportedApi when
        // SHADER_F64 is absent from the feature list.
        //
        // This function is factored out of check_f64_capability so that the
        // SHADER_F64 gate is testable without requiring GPU hardware.
        let features_without_f64: Vec<String> =
            vec!["TIMESTAMP_QUERY".to_owned(), "PUSH_CONSTANTS".to_owned()];
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
        let features_with_f64: Vec<String> =
            vec!["SHADER_F64".to_owned(), "TIMESTAMP_QUERY".to_owned()];
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
        // Staging must be sized to the full spinor output: for shells l=1 (nso=6) and l=2 (nso=10),
        // n_elem = 6*10 = 60 for the result, but spinor stores real+imag interleaved so staging
        // needs 2*n_elem = 120 f64s. (The CR-01 BufferTooSmall guard in 20-10 now enforces this.)
        let spinor_basis = Box::leak(Box::new(sample_basis(Representation::Spinor, 2)));
        let spinor_plan = build_plan(spinor_basis, 2, Representation::Spinor, 2);
        let spinor_chunk = spinor_plan.workspace.chunks[0].clone();
        let spinor_staging_elems = spinor_plan.output_layout.staging_elements.max(8);
        let mut spinor_staging = vec![0.0_f64; spinor_staging_elems];
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
