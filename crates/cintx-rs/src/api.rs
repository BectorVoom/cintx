//! Safe Rust facade scaffolding for query/evaluate session flows.

use crate::error::FacadeError;
use cintx_compat::raw::enforce_safe_facade_policy_gate;
use cintx_core::{BasisSet, CintFloat, OperatorId, PrecisionKind, Representation, ShellTuple};
use cintx_cubecl::{CubeClExecutor, EriSsssInput, OverlapSsInput};
use cintx_ops::resolver::Resolver;
use cintx_runtime::{
    BackendExecutor, BatchExecutionPlan, BatchItemRequest, ExecutionIo, ExecutionOptions,
    ExecutionPlan, ExecutionStats, KernelClass, ReusableWorkspaceAllocator, WorkspaceAllocator,
    WorkspaceQuery as RuntimeWorkspaceQuery, query_workspace as runtime_query_workspace,
    schedule_chunks,
};
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Reusable safe-API execution state.
///
/// A context owns the backend-client cache and a reusable host workspace without exposing
/// CubeCL types in the public API. Reuse one context for related requests to avoid repeated
/// backend bootstrap and host scratch allocation. Device-resident basis uploads are not yet
/// implemented; the executor currently retains content-addressed basis metadata only.
#[derive(Clone, Debug)]
pub struct EvaluationContext {
    executor: Arc<CubeClExecutor>,
    workspace_allocator: Arc<Mutex<ReusableWorkspaceAllocator>>,
}

/// Measured reusable state held by an [`EvaluationContext`].
///
/// `resident_metadata_entries` counts content-addressed basis metadata. It is not a
/// device-upload count until persistent device buffers are implemented.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationContextStats {
    pub backend_cache_entries: usize,
    pub resident_metadata_entries: usize,
    pub host_workspace_allocations: usize,
    pub host_workspace_reuses: usize,
    pub host_workspace_peak_bytes: usize,
    /// Total output-only device staging allocations made by the verified
    /// Cartesian s-s batch pilot. Dynamic descriptor tables are uploaded for
    /// each request and are not counted as reusable device residency.
    pub pilot_output_staging_allocations: usize,
    /// Total same-or-smaller pilot chunks served by retained output storage.
    pub pilot_output_staging_reuses: usize,
    /// Total retained output slots replaced because a larger chunk was needed.
    pub pilot_output_staging_growths: usize,
    /// Bytes currently retained in the output-only pilot arena.
    pub pilot_output_staging_retained_bytes: usize,
    /// High-water mark for retained output-only pilot staging bytes.
    pub pilot_output_staging_peak_bytes: usize,
}

impl EvaluationContext {
    pub fn new() -> Self {
        Self {
            executor: Arc::new(CubeClExecutor::new()),
            workspace_allocator: Arc::new(Mutex::new(ReusableWorkspaceAllocator::default())),
        }
    }

    pub fn stats(&self) -> EvaluationContextStats {
        let arena = self
            .workspace_allocator
            .lock()
            .expect("evaluation context workspace arena poisoned");
        let pilot_arena = self.executor.pilot_output_arena_stats();
        EvaluationContextStats {
            backend_cache_entries: self.executor.backend_cache_entries(),
            resident_metadata_entries: self.executor.resident_cache().len(),
            host_workspace_allocations: arena.allocations(),
            host_workspace_reuses: arena.reuses(),
            host_workspace_peak_bytes: arena.peak_bytes(),
            pilot_output_staging_allocations: pilot_arena.allocations,
            pilot_output_staging_reuses: pilot_arena.reuses,
            pilot_output_staging_growths: pilot_arena.growths,
            pilot_output_staging_retained_bytes: pilot_arena.retained_bytes,
            pilot_output_staging_peak_bytes: pilot_arena.peak_retained_bytes,
        }
    }
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Reusable safe-API execution session.
///
/// A session owns the backend-client cache and host workspace arena used by its
/// queries. It exposes no CubeCL types; callers supply ordinary
/// [`SessionRequest`] or [`BatchRequest`] values and can inspect only the
/// portable [`EvaluationContextStats`] counters. Clones intentionally share
/// the same retained context, which makes it suitable for handing the safe
/// facade to adjacent planning code without creating another backend client.
#[derive(Clone, Debug, Default)]
pub struct Session {
    context: EvaluationContext,
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspect reusable host/backend state accumulated by this session.
    pub fn stats(&self) -> EvaluationContextStats {
        self.context.stats()
    }

    /// Validate and plan a request using this session's retained context.
    pub fn query<'basis>(
        &self,
        request: &SessionRequest<'basis>,
    ) -> Result<SessionQuery<'basis>, FacadeError> {
        request.query_workspace_in(&self.context)
    }

    /// Evaluate a single request using this session's retained context.
    pub fn evaluate<'basis>(
        &self,
        request: &SessionRequest<'basis>,
    ) -> Result<TypedEvaluationOutput<f64>, FacadeError> {
        self.query(request)?.evaluate()
    }

    /// Transactionally evaluate a batch using this session's retained context.
    pub fn evaluate_batch<'basis>(
        &self,
        batch: BatchRequest<'basis>,
    ) -> Result<BatchEvaluationOutput<f64>, FacadeError> {
        batch.evaluate_batch_in(&self.context)
    }

    /// Transactionally commit a batch into caller-owned output storage.
    pub fn evaluate_batch_into<'basis>(
        &self,
        batch: BatchRequest<'basis>,
        destination: &mut [TypedEvaluationOutput<f64>],
    ) -> Result<BatchExecutionPlan, FacadeError> {
        batch.evaluate_batch_into_in(&self.context, destination)
    }
}

/// Typed safe request object that keeps `query_workspace()` and `evaluate()` connected.
#[derive(Clone, Debug)]
pub struct SessionRequest<'basis> {
    operator: OperatorId,
    representation: Representation,
    basis: &'basis BasisSet,
    shells: ShellTuple,
    options: ExecutionOptions,
}

/// A safe multi-request submission with transactional caller-visible results.
///
/// The plan is built before evaluation, so invalid requests and impossible
/// chunk limits fail without writing an `evaluate_batch_into` destination.
/// Current kernels use the scalar compatibility route per planned item; the
/// public contract and offsets are already the same ones consumed by the
/// batched CubeCL launchers.
#[derive(Clone, Debug)]
pub struct BatchRequest<'basis> {
    requests: Vec<SessionRequest<'basis>>,
    max_items_per_chunk: usize,
    max_chunk_bytes: usize,
}

/// Results from a batch request, ordered exactly like its input requests.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchEvaluationOutput<F = f64> {
    pub plan: BatchExecutionPlan,
    /// Submission-level counters for the complete transactional batch.
    pub stats: BatchExecutionStats,
    pub outputs: Vec<TypedEvaluationOutput<F>>,
}

/// Counters that describe one complete batch submission rather than one item.
///
/// The counters are intentionally separate from [`EvaluationStats`]: one batch
/// may contain several launches but only one collective readback. This makes
/// launch-amortization measurements attributable without changing scalar API
/// semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchExecutionStats {
    pub items_planned: usize,
    pub items_executed: usize,
    pub bucket_count: usize,
    pub chunk_count: usize,
    pub kernel_launch_count: usize,
    pub readback_count: usize,
    pub transfer_bytes: usize,
    /// Host time spent arranging the planned descriptors into chunk-local tables.
    pub pack_ns: u64,
    /// Host time spent creating buffers and encoding all pilot launches.
    ///
    /// This includes host-to-device work where the backend performs it eagerly;
    /// it is not a device kernel timestamp.
    pub submit_ns: u64,
    /// Host time in the pilot's one collective readback/synchronization call.
    /// This is not a device transfer timestamp.
    pub readback_ns: u64,
    /// Output-only pilot staging allocations made during this batch.
    pub pilot_output_staging_allocations: usize,
    /// Output chunks that reused a same-or-larger retained staging slot.
    pub pilot_output_staging_reuses: usize,
    /// Output slots replaced because the current chunk exceeded capacity.
    pub pilot_output_staging_growths: usize,
}

impl<'basis> BatchRequest<'basis> {
    pub fn new(requests: impl IntoIterator<Item = SessionRequest<'basis>>) -> Self {
        Self {
            requests: requests.into_iter().collect(),
            max_items_per_chunk: usize::MAX,
            max_chunk_bytes: usize::MAX,
        }
    }

    pub fn max_items_per_chunk(mut self, value: usize) -> Self {
        self.max_items_per_chunk = value.max(1);
        self
    }

    pub fn max_chunk_bytes(mut self, value: usize) -> Self {
        self.max_chunk_bytes = value.max(1);
        self
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Evaluate every item and return results only after the whole batch succeeds.
    pub fn evaluate_batch(self) -> Result<BatchEvaluationOutput<f64>, FacadeError> {
        self.evaluate_batch_in(&EvaluationContext::new())
    }

    /// Evaluate with a reusable context that retains backend and resident-basis state.
    pub fn evaluate_batch_in(
        self,
        context: &EvaluationContext,
    ) -> Result<BatchEvaluationOutput<f64>, FacadeError> {
        let mut queries = Vec::with_capacity(self.requests.len());
        let mut items = Vec::with_capacity(self.requests.len());
        for request in &self.requests {
            let query = request.query_workspace_in(context)?;
            let descriptor = Resolver::descriptor(request.operator).map_err(|error| {
                FacadeError::UnsupportedApi {
                    requested: error.to_string(),
                }
            })?;
            let plan = ExecutionPlan::new(
                request.operator,
                request.representation,
                request.basis,
                request.shells.clone(),
                &query.runtime_workspace,
            )
            .map_err(FacadeError::from)?;
            items.push(BatchItemRequest {
                kernel_class: KernelClass {
                    family: descriptor.family(),
                    representation: request.representation,
                    precision: request.options.precision,
                    arity: request.shells.len() as u8,
                    angular_momenta: request
                        .shells
                        .iter()
                        .map(|shell| shell.ang_momentum)
                        .collect(),
                    nroots: 0,
                    component_rank: plan.component_count.min(u8::MAX as usize) as u8,
                },
                output_elements: plan.output_layout.staging_elements,
                scratch_bytes: query.runtime_workspace.bytes,
            });
            queries.push(query);
        }
        let plan = BatchExecutionPlan::build(items, self.max_items_per_chunk, self.max_chunk_bytes)
            .map_err(FacadeError::from)?;

        if let Some((intent, inputs, kinetic)) = ss_batch_inputs(&self.requests) {
            let execution = execute_ss_batch_chunks(context, &plan, &intent, &inputs, kinetic)?;
            let stats = BatchExecutionStats {
                items_planned: plan.items.len(),
                items_executed: execution.values.len(),
                bucket_count: plan.buckets.len(),
                chunk_count: plan.chunks.len(),
                kernel_launch_count: plan.chunks.len(),
                readback_count: usize::from(!execution.values.is_empty()),
                transfer_bytes: execution.per_item_transfer_bytes.iter().sum(),
                pack_ns: execution.pack_ns,
                submit_ns: execution.submit_ns,
                readback_ns: execution.readback_ns,
                pilot_output_staging_allocations: execution.output_staging_allocations,
                pilot_output_staging_reuses: execution.output_staging_reuses,
                pilot_output_staging_growths: execution.output_staging_growths,
            };
            let outputs = execution
                .values
                .into_iter()
                .zip(execution.per_item_transfer_bytes)
                .map(|(value, transfer_bytes)| overlap_ss_batch_output(value, transfer_bytes))
                .collect();
            return Ok(BatchEvaluationOutput {
                plan,
                stats,
                outputs,
            });
        }

        if let Some((intent, inputs)) = eri_ssss_batch_inputs(&self.requests) {
            let execution = execute_eri_ssss_batch_chunks(context, &plan, &intent, &inputs)?;
            let stats = BatchExecutionStats {
                items_planned: plan.items.len(),
                items_executed: execution.values.len(),
                bucket_count: plan.buckets.len(),
                chunk_count: plan.chunks.len(),
                kernel_launch_count: plan.chunks.len(),
                readback_count: usize::from(!execution.values.is_empty()),
                transfer_bytes: execution.per_item_transfer_bytes.iter().sum(),
                pack_ns: execution.pack_ns,
                submit_ns: execution.submit_ns,
                readback_ns: execution.readback_ns,
                pilot_output_staging_allocations: 0,
                pilot_output_staging_reuses: 0,
                pilot_output_staging_growths: 0,
            };
            let outputs = execution
                .values
                .into_iter()
                .zip(execution.per_item_transfer_bytes)
                .map(|(value, transfer_bytes)| eri_ssss_batch_output(value, transfer_bytes))
                .collect();
            return Ok(BatchEvaluationOutput {
                plan,
                stats,
                outputs,
            });
        }

        // Keep the compatibility route transactional while the per-class
        // batched launchers are migrated: no destination is exposed until all
        // planned items have completed.
        let mut outputs = Vec::with_capacity(queries.len());
        for query in queries {
            outputs.push(query.evaluate()?);
        }
        let stats = BatchExecutionStats {
            items_planned: plan.items.len(),
            items_executed: outputs.len(),
            bucket_count: plan.buckets.len(),
            chunk_count: plan.chunks.len(),
            kernel_launch_count: outputs.iter().map(|output| output.stats.chunk_count).sum(),
            readback_count: outputs.iter().map(|output| output.stats.chunk_count).sum(),
            transfer_bytes: outputs
                .iter()
                .map(|output| output.stats.transfer_bytes)
                .sum(),
            pack_ns: 0,
            submit_ns: 0,
            readback_ns: 0,
            pilot_output_staging_allocations: 0,
            pilot_output_staging_reuses: 0,
            pilot_output_staging_growths: 0,
        };
        Ok(BatchEvaluationOutput {
            plan,
            stats,
            outputs,
        })
    }

    /// Transactionally commit a batch into a caller-provided result slice.
    pub fn evaluate_batch_into(
        self,
        destination: &mut [TypedEvaluationOutput<f64>],
    ) -> Result<BatchExecutionPlan, FacadeError> {
        self.evaluate_batch_into_in(&EvaluationContext::new(), destination)
    }

    /// Transactionally commit a context-backed batch into a caller-provided result slice.
    pub fn evaluate_batch_into_in(
        self,
        context: &EvaluationContext,
        destination: &mut [TypedEvaluationOutput<f64>],
    ) -> Result<BatchExecutionPlan, FacadeError> {
        if destination.len() != self.requests.len() {
            return Err(FacadeError::Layout {
                detail: format!(
                    "batch destination length mismatch: expected {}, provided {}",
                    self.requests.len(),
                    destination.len()
                ),
            });
        }
        let batch = self.evaluate_batch_in(context)?;
        for (slot, output) in destination.iter_mut().zip(batch.outputs) {
            *slot = output;
        }
        Ok(batch.plan)
    }
}

/// Execute the overlap pilot according to the plan's real, disjoint item ranges.
///
/// The staging vector is private until every chunk succeeds. This keeps chunking
/// computational rather than merely advisory and retains the safe API's
/// transactional result contract.
struct SsBatchExecution {
    values: Vec<f64>,
    per_item_transfer_bytes: Vec<usize>,
    pack_ns: u64,
    submit_ns: u64,
    readback_ns: u64,
    output_staging_allocations: usize,
    output_staging_reuses: usize,
    output_staging_growths: usize,
}

fn execute_ss_batch_chunks(
    context: &EvaluationContext,
    plan: &BatchExecutionPlan,
    intent: &cintx_runtime::BackendIntent,
    inputs: &[OverlapSsInput],
    kinetic: bool,
) -> Result<SsBatchExecution, FacadeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(inputs.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} scalar batch result elements",
                inputs.len()
            ),
        })?;
    values.resize(inputs.len(), 0.0);
    let mut per_item_transfer_bytes = vec![0; inputs.len()];

    let mut executed = Vec::new();
    executed
        .try_reserve_exact(inputs.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} scalar batch execution markers",
                inputs.len()
            ),
        })?;
    executed.resize(inputs.len(), false);

    let mut packed_chunks = Vec::new();
    packed_chunks
        .try_reserve_exact(plan.chunks.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} scalar batch chunk descriptors",
                plan.chunks.len()
            ),
        })?;
    let pack_started = Instant::now();
    for chunk in &plan.chunks {
        let item_indices = plan.chunk_items(chunk);
        let mut chunk_inputs = Vec::new();
        chunk_inputs
            .try_reserve_exact(item_indices.len())
            .map_err(|_| FacadeError::Memory {
                detail: format!(
                    "failed to allocate {} scalar batch chunk descriptors",
                    item_indices.len()
                ),
            })?;
        for &item_index in item_indices {
            let input = inputs
                .get(item_index)
                .cloned()
                .ok_or_else(|| FacadeError::Validation {
                    detail: format!(
                        "scalar batch plan referenced input {item_index}, but only {} inputs exist",
                        inputs.len()
                    ),
                })?;
            chunk_inputs.push(input);
        }
        packed_chunks.push(chunk_inputs);
    }

    let chunk_refs: Vec<_> = packed_chunks.iter().map(Vec::as_slice).collect();
    let pack_ns = elapsed_ns(pack_started);
    let submission = context
        .executor
        .execute_ss_batch_chunks(intent, &chunk_refs, kinetic)
        .map_err(FacadeError::from)?;
    let chunk_results = submission.chunks;
    let chunk_transfer_bytes = submission.chunk_transfer_bytes;
    if chunk_results.len() != plan.chunks.len() {
        return Err(FacadeError::Validation {
            detail: format!(
                "scalar batch chunk result count mismatch: expected {}, provided {}",
                plan.chunks.len(),
                chunk_results.len()
            ),
        });
    }
    if chunk_transfer_bytes.len() != plan.chunks.len() {
        return Err(FacadeError::Validation {
            detail: format!(
                "scalar batch transfer count mismatch: expected {}, provided {}",
                plan.chunks.len(),
                chunk_transfer_bytes.len()
            ),
        });
    }

    for ((chunk, chunk_values), chunk_bytes) in plan
        .chunks
        .iter()
        .zip(chunk_results)
        .zip(chunk_transfer_bytes)
    {
        let item_indices = plan.chunk_items(chunk);
        if chunk_values.len() != item_indices.len() {
            return Err(FacadeError::Validation {
                detail: format!(
                    "scalar batch chunk {} result length mismatch: expected {}, provided {}",
                    chunk.index,
                    item_indices.len(),
                    chunk_values.len()
                ),
            });
        }
        let bytes_per_item = chunk_bytes / item_indices.len();
        if bytes_per_item.saturating_mul(item_indices.len()) != chunk_bytes {
            return Err(FacadeError::Validation {
                detail: format!(
                    "scalar batch chunk {} transfer bytes are not item-aligned",
                    chunk.index
                ),
            });
        }

        for (&item_index, value) in item_indices.iter().zip(chunk_values) {
            let output_len = values.len();
            let slot = values
                .get_mut(item_index)
                .ok_or_else(|| FacadeError::Validation {
                    detail: format!(
                        "scalar batch plan wrote output {item_index}, but only {output_len} output slots exist"
                    ),
                })?;
            let was_executed = executed
                .get_mut(item_index)
                .expect("input and execution marker lengths are identical");
            if *was_executed {
                return Err(FacadeError::Validation {
                    detail: format!("scalar batch plan executed input {item_index} more than once"),
                });
            }
            *slot = value;
            per_item_transfer_bytes[item_index] = bytes_per_item;
            *was_executed = true;
        }
    }

    if let Some((item_index, _)) = executed.iter().enumerate().find(|(_, done)| !**done) {
        return Err(FacadeError::Validation {
            detail: format!("scalar batch plan did not execute input {item_index}"),
        });
    }
    Ok(SsBatchExecution {
        values,
        per_item_transfer_bytes,
        pack_ns,
        submit_ns: submission.submit_ns,
        readback_ns: submission.readback_ns,
        output_staging_allocations: submission.output_staging_allocations,
        output_staging_reuses: submission.output_staging_reuses,
        output_staging_growths: submission.output_staging_growths,
    })
}

/// Execute primitive `(s s | s s)` chunks while retaining private staging
/// until every chunk has completed.  This mirrors the 1e pilot's transactional
/// contract but remains a distinct 2e path and descriptor type.
fn execute_eri_ssss_batch_chunks(
    context: &EvaluationContext,
    plan: &BatchExecutionPlan,
    intent: &cintx_runtime::BackendIntent,
    inputs: &[EriSsssInput],
) -> Result<SsBatchExecution, FacadeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(inputs.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} primitive ERI batch result elements",
                inputs.len()
            ),
        })?;
    values.resize(inputs.len(), 0.0);
    let mut per_item_transfer_bytes = Vec::new();
    per_item_transfer_bytes
        .try_reserve_exact(inputs.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} primitive ERI transfer counters",
                inputs.len()
            ),
        })?;
    per_item_transfer_bytes.resize(inputs.len(), 0);
    let mut executed = Vec::new();
    executed
        .try_reserve_exact(inputs.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} primitive ERI execution markers",
                inputs.len()
            ),
        })?;
    executed.resize(inputs.len(), false);
    let mut packed_chunks = Vec::new();
    packed_chunks
        .try_reserve_exact(plan.chunks.len())
        .map_err(|_| FacadeError::Memory {
            detail: format!(
                "failed to allocate {} primitive ERI batch chunk descriptors",
                plan.chunks.len()
            ),
        })?;
    let pack_started = Instant::now();
    for chunk in &plan.chunks {
        let item_indices = plan.chunk_items(chunk);
        let mut chunk_inputs = Vec::new();
        chunk_inputs
            .try_reserve_exact(item_indices.len())
            .map_err(|_| FacadeError::Memory {
                detail: format!(
                    "failed to allocate {} primitive ERI chunk descriptors",
                    item_indices.len()
                ),
            })?;
        for &item_index in item_indices {
            chunk_inputs.push(inputs.get(item_index).cloned().ok_or_else(|| {
                FacadeError::Validation {
                    detail: format!(
                        "primitive ERI batch plan referenced input {item_index}, but only {} inputs exist",
                        inputs.len()
                    ),
                }
            })?);
        }
        packed_chunks.push(chunk_inputs);
    }
    let chunk_refs: Vec<_> = packed_chunks.iter().map(Vec::as_slice).collect();
    let pack_ns = elapsed_ns(pack_started);
    let submission = context
        .executor
        .execute_eri_ssss_batch_chunks(intent, &chunk_refs)
        .map_err(FacadeError::from)?;
    if submission.chunks.len() != plan.chunks.len()
        || submission.chunk_transfer_bytes.len() != plan.chunks.len()
    {
        return Err(FacadeError::Validation {
            detail: "primitive ERI batch submission did not return one result and transfer count per planned chunk".to_owned(),
        });
    }
    for ((chunk, chunk_values), chunk_bytes) in plan
        .chunks
        .iter()
        .zip(submission.chunks)
        .zip(submission.chunk_transfer_bytes)
    {
        let item_indices = plan.chunk_items(chunk);
        if chunk_values.len() != item_indices.len() || chunk_bytes % item_indices.len().max(1) != 0
        {
            return Err(FacadeError::Validation {
                detail: format!(
                    "primitive ERI batch chunk {} violated its output or transfer layout",
                    chunk.index
                ),
            });
        }
        let bytes_per_item = chunk_bytes / item_indices.len();
        for (&item_index, value) in item_indices.iter().zip(chunk_values) {
            let slot = values
                .get_mut(item_index)
                .ok_or_else(|| FacadeError::Validation {
                    detail: format!("primitive ERI batch wrote output {item_index} out of bounds"),
                })?;
            let was_executed = executed.get_mut(item_index).expect("matched marker length");
            if *was_executed {
                return Err(FacadeError::Validation {
                    detail: format!(
                        "primitive ERI batch executed input {item_index} more than once"
                    ),
                });
            }
            *slot = value;
            per_item_transfer_bytes[item_index] = bytes_per_item;
            *was_executed = true;
        }
    }
    if let Some((item_index, _)) = executed.iter().enumerate().find(|(_, done)| !**done) {
        return Err(FacadeError::Validation {
            detail: format!("primitive ERI batch did not execute input {item_index}"),
        });
    }
    Ok(SsBatchExecution {
        values,
        per_item_transfer_bytes,
        pack_ns,
        submit_ns: submission.submit_ns,
        readback_ns: submission.readback_ns,
        output_staging_allocations: 0,
        output_staging_reuses: 0,
        output_staging_growths: 0,
    })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Build the descriptor table for the first safe-API batch kernels.
///
/// The fast paths are deliberately exact: `int1e_ovlp_cart` and `int1e_kin_cart`, two
/// Cartesian `s` shells, arbitrary primitive counts and one contraction on each shell, and f64
/// precision. Keeping this predicate narrow preserves the established compatibility
/// implementation for multi-contraction, angular, derivative, and non-overlap requests
/// until their batch kernels carry equivalent descriptors and output transforms.
fn ss_batch_inputs(
    requests: &[SessionRequest<'_>],
) -> Option<(cintx_runtime::BackendIntent, Vec<OverlapSsInput>, bool)> {
    let first = requests.first()?;
    let intent = first.options.backend_intent.clone();
    let mut inputs = Vec::with_capacity(requests.len());

    for request in requests {
        if !matches!(request.operator.raw(), 0 | 3)
            || request.representation != Representation::Cart
            || request.options.precision != PrecisionKind::F64
            || request.options.backend_intent != intent
        {
            return None;
        }
        let [shell_i, shell_j] = request.shells.as_slice() else {
            return None;
        };
        if shell_i.ang_momentum != 0
            || shell_j.ang_momentum != 0
            || shell_i.nctr != 1
            || shell_j.nctr != 1
            || shell_i.representation != Representation::Cart
            || shell_j.representation != Representation::Cart
        {
            return None;
        }
        let atom_i = request.basis.atoms().get(shell_i.atom_index as usize)?;
        let atom_j = request.basis.atoms().get(shell_j.atom_index as usize)?;
        inputs.push(OverlapSsInput {
            exponents_i: Arc::clone(&shell_i.exponents),
            exponents_j: Arc::clone(&shell_j.exponents),
            coefficients_i: Arc::clone(&shell_i.coefficients),
            coefficients_j: Arc::clone(&shell_j.coefficients),
            center_i: atom_i.coord_bohr,
            center_j: atom_j.coord_bohr,
        });
    }

    let kinetic = first.operator.raw() == 3;
    if requests
        .iter()
        .any(|request| (request.operator.raw() == 3) != kinetic)
    {
        return None;
    }
    Some((intent, inputs, kinetic))
}

/// Build descriptors for the primitive Cartesian `int2e_cart` pilot only.
///
/// The predicate is intentionally narrower than the 1e pilot: all four
/// shells must be uncontracted s shells with exactly one primitive.  Any
/// other two-electron input keeps the scalar compatibility execution path.
fn eri_ssss_batch_inputs(
    requests: &[SessionRequest<'_>],
) -> Option<(cintx_runtime::BackendIntent, Vec<EriSsssInput>)> {
    let first = requests.first()?;
    let intent = first.options.backend_intent.clone();
    let mut inputs = Vec::with_capacity(requests.len());

    for request in requests {
        if Resolver::descriptor(request.operator)
            .ok()?
            .entry
            .symbol_name
            != "int2e_cart"
            || request.representation != Representation::Cart
            || request.options.precision != PrecisionKind::F64
            || request.options.backend_intent != intent
        {
            return None;
        }
        let [shell_a, shell_b, shell_c, shell_d] = request.shells.as_slice() else {
            return None;
        };
        for shell in [shell_a, shell_b, shell_c, shell_d] {
            if shell.ang_momentum != 0
                || shell.nprim != 1
                || shell.nctr != 1
                || shell.representation != Representation::Cart
                || shell.exponents.len() != 1
                || shell.coefficients.len() != 1
            {
                return None;
            }
        }
        let center_a = request
            .basis
            .atoms()
            .get(shell_a.atom_index as usize)?
            .coord_bohr;
        let center_b = request
            .basis
            .atoms()
            .get(shell_b.atom_index as usize)?
            .coord_bohr;
        let center_c = request
            .basis
            .atoms()
            .get(shell_c.atom_index as usize)?
            .coord_bohr;
        let center_d = request
            .basis
            .atoms()
            .get(shell_d.atom_index as usize)?
            .coord_bohr;
        inputs.push(EriSsssInput {
            exponents: [
                shell_a.exponents[0],
                shell_b.exponents[0],
                shell_c.exponents[0],
                shell_d.exponents[0],
            ],
            coefficients: [
                shell_a.coefficients[0],
                shell_b.coefficients[0],
                shell_c.coefficients[0],
                shell_d.coefficients[0],
            ],
            centers: [center_a, center_b, center_c, center_d],
        });
    }
    Some((intent, inputs))
}

fn overlap_ss_batch_output(value: f64, transfer_bytes: usize) -> TypedEvaluationOutput<f64> {
    const OUTPUT_BYTES_PER_ITEM: usize = size_of::<f64>();
    TypedEvaluationOutput {
        tensor: IntegralTensor {
            extents: vec![1, 1],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![value],
        },
        stats: EvaluationStats {
            chunk_count: 1,
            planned_batches: 1,
            transfer_bytes,
            not0: i32::from(value != 0.0),
            ..EvaluationStats::default()
        },
        workspace_bytes: 0,
        chunk_count: 1,
        bytes_written: OUTPUT_BYTES_PER_ITEM,
    }
}

fn eri_ssss_batch_output(value: f64, transfer_bytes: usize) -> TypedEvaluationOutput<f64> {
    TypedEvaluationOutput {
        tensor: IntegralTensor {
            extents: vec![1, 1, 1, 1],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![value],
        },
        stats: EvaluationStats {
            chunk_count: 1,
            planned_batches: 1,
            transfer_bytes,
            not0: i32::from(value != 0.0),
            ..EvaluationStats::default()
        },
        workspace_bytes: 0,
        chunk_count: 1,
        bytes_written: size_of::<f64>(),
    }
}

impl<'basis> SessionRequest<'basis> {
    pub fn new(
        operator: OperatorId,
        representation: Representation,
        basis: &'basis BasisSet,
        shells: ShellTuple,
        options: ExecutionOptions,
    ) -> Self {
        Self {
            operator,
            representation,
            basis,
            shells,
            options,
        }
    }

    pub fn operator(&self) -> OperatorId {
        self.operator
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn basis(&self) -> &'basis BasisSet {
        self.basis
    }

    pub fn shells(&self) -> &ShellTuple {
        &self.shells
    }

    pub fn options(&self) -> &ExecutionOptions {
        &self.options
    }

    pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
        self.query_workspace_in(&EvaluationContext::new())
    }

    /// Validate and plan this request using a reusable execution context.
    pub fn query_workspace_in(
        &self,
        context: &EvaluationContext,
    ) -> Result<SessionQuery<'basis>, FacadeError> {
        // Phase 18 D-04: aosym preflight — only S1 (and None ≡ S1) is implemented.
        // Non-S1 packings return a typed FacadeError::UnsupportedAoSymmetry so callers
        // can pattern-match programmatically. Fails fast before any runtime work.
        if let Some(aosym) = self.options.aosym {
            if aosym != cintx_core::AoSymmetry::S1 {
                return Err(FacadeError::UnsupportedAoSymmetry {
                    requested: aosym.to_string(),
                });
            }
        }

        // Phase 19 D-06: ECP-basis preflight — operator.is_ecp() &&
        // basis.ecp_shells().is_empty() returns FacadeError::MissingEcpBasis.
        // Fails fast before any runtime_query_workspace work, mirroring the
        // aosym preflight above. Resolves the operator symbol via the
        // manifest so the error message names the canonical libcint symbol;
        // falls back to the OperatorId Display impl if the manifest lookup
        // fails (defensive — keeps the safe API from panicking on a missing
        // descriptor).
        if self.operator.is_ecp() && self.basis.ecp_shells().is_empty() {
            let symbol = Resolver::descriptor(self.operator)
                .map(|d| d.operator_symbol().to_string())
                .unwrap_or_else(|_| format!("{}", self.operator));
            return Err(FacadeError::MissingEcpBasis { operator: symbol });
        }

        let runtime_workspace = runtime_query_workspace(
            self.operator,
            self.representation,
            self.basis,
            self.shells.clone(),
            &self.options,
        )
        .map_err(FacadeError::from)?;

        let workspace = WorkspacePlan::from_runtime(self, &runtime_workspace);
        Ok(SessionQuery {
            request: self.clone(),
            workspace,
            runtime_workspace,
            executor: Arc::clone(&context.executor),
            workspace_allocator: Arc::clone(&context.workspace_allocator),
        })
    }
}

/// Result of `query_workspace()` that carries the validated request metadata forward to evaluate.
#[derive(Clone, Debug)]
pub struct SessionQuery<'basis> {
    request: SessionRequest<'basis>,
    workspace: WorkspacePlan,
    runtime_workspace: RuntimeWorkspaceQuery,
    executor: Arc<CubeClExecutor>,
    workspace_allocator: Arc<Mutex<ReusableWorkspaceAllocator>>,
}

impl<'basis> SessionQuery<'basis> {
    pub fn request(&self) -> &SessionRequest<'basis> {
        &self.request
    }

    pub fn workspace(&self) -> &WorkspacePlan {
        &self.workspace
    }

    /// Evaluate the integral using the default f64 precision.
    ///
    /// This is a thin shim that delegates to `evaluate_generic::<f64>()`. It exists
    /// so that every existing `req.evaluate()` call site compiles unchanged without
    /// a turbofish (D-03/D-12). The result is byte-identical to the pre-generic
    /// implementation.
    pub fn evaluate(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> {
        self.evaluate_generic::<f64>()
    }

    /// Evaluate the integral using the precision specified by the type parameter `F`.
    ///
    /// - `evaluate_generic::<f64>()` — byte-identical to the pre-generic `evaluate()` (D-12).
    /// - `evaluate_generic::<f32>()` — opts into f32 output; sets `plan.precision = F32`;
    ///   returns `Vec<f32>` (T-20-19 type-system isolation).
    ///
    /// # Staging buffer design
    ///
    /// The facade owns a `Vec<F>` sized in `size_of::<F>()` per element. When passing to
    /// `ExecutionIo::new` (which requires `&mut [f64]` — frozen interface), the buffer is
    /// reinterpreted via `bytemuck::cast_slice_mut::<F, f64>`. This is sound because both
    /// `f32` and `f64` are `bytemuck::Pod` (proven in Plan 01, A5 spike), and the kernel
    /// dispatcher reads back the same byte pattern it wrote (it branches on `plan.precision`).
    ///
    /// OOM-safe fallible allocation (`try_reserve_exact`) and the no-partial-writes contract
    /// are preserved verbatim.
    pub fn evaluate_generic<F: CintFloat + bytemuck::Pod>(
        self,
    ) -> Result<TypedEvaluationOutput<F>, FacadeError> {
        self.workspace
            .execution_token
            .ensure_matches(&self.request, &self.runtime_workspace)?;

        let descriptor = Resolver::descriptor(self.request.operator).map_err(|err| {
            FacadeError::UnsupportedApi {
                requested: err.to_string(),
            }
        })?;
        // Preflight source/profile/optional policy before ExecutionPlan::new so source-only
        // operators fail with compat-origin UnsupportedApi reasons instead of planner internals.
        enforce_safe_facade_policy_gate(
            descriptor,
            self.request.representation,
            &self.request.shells,
            &[],
        )
        .map_err(FacadeError::from)?;

        let mut plan = ExecutionPlan::new(
            self.request.operator,
            self.request.representation,
            self.request.basis,
            self.request.shells.clone(),
            &self.runtime_workspace,
        )
        .map_err(FacadeError::from)?;

        // Set precision from the F type parameter (Plan 07 wiring note):
        // F::PRECISION maps to PrecisionKind::F64 for f64 and PrecisionKind::F32 for f32.
        // This must happen before any planner::evaluate call so kernel dispatchers (Plans 04/05)
        // pick the right monomorphization. Follows the f12_zeta caller-populates-after-new
        // precedent established in Plan 06.
        plan.precision = F::PRECISION;

        // Propagate f12_zeta from ExecutionOptions to operator_env_params (safe API path).
        if let Some(zeta) = self.request.options().f12_zeta {
            plan.operator_env_params.f12_zeta = Some(zeta);
        }
        // Propagate rinv_orig from ExecutionOptions to operator_env_params (safe API path, Plan 21-01).
        if let Some(origin) = self.request.options().rinv_orig {
            plan.operator_env_params.rinv_orig = Some(origin);
        }
        // Propagate common_orig from ExecutionOptions to operator_env_params (safe API path, Plan 22-01).
        if let Some(origin) = self.request.options().common_orig {
            plan.operator_env_params.common_orig = Some(origin);
        }
        // Phase 22 FND-01 (gap closure): finiteness-validate the gauge origin on the safe-API
        // path too. Mirrors the raw-path guard in cintx-compat raw.rs so the builder doc
        // contract ("NaN/inf rejected by validate_common_orig_env_params") holds on BOTH paths
        // — a `.with_common_origin([NaN, ..])` caller now gets InvalidEnvParam, not a silent
        // garbage origin threaded into the plan.
        cintx_runtime::validator::validate_common_orig_env_params(
            plan.descriptor.operator_name(),
            &plan.operator_env_params,
        )
        .map_err(FacadeError::from)?;

        enforce_safe_facade_policy_gate(
            plan.descriptor,
            self.request.representation,
            &self.request.shells,
            &plan.output_layout.extents,
        )
        .map_err(FacadeError::from)?;

        let output_layout = plan.output_layout.clone();
        // A context serializes host scratch reuse. This avoids repeated allocations
        // while preserving the fallible, no-partial-write allocation boundary.
        let mut allocator = self
            .workspace_allocator
            .lock()
            .expect("evaluation context workspace arena poisoned");
        let executor = &self.executor;

        if !executor.supports(&plan) {
            return Err(FacadeError::UnsupportedApi {
                requested: format!(
                    "{}/{}/{}",
                    plan.descriptor.family(),
                    plan.descriptor.operator_name(),
                    self.request.representation
                ),
            });
        }

        let backend_workspace = executor
            .query_workspace(&plan)
            .map_err(FacadeError::from)?
            .get();
        if backend_workspace > plan.workspace.bytes {
            return Err(FacadeError::from(
                cintx_core::cintxRsError::MemoryLimitExceeded {
                    requested: backend_workspace,
                    limit: plan.workspace.bytes,
                },
            ));
        }

        // Allocate the output accumulator as Vec<F>.
        // Buffer is sized in elements; each element is size_of::<F>() bytes (T-20-20 mitigation).
        let staging_elements = output_layout.staging_elements;
        let staging_bytes =
            staging_elements
                .checked_mul(size_of::<F>())
                .ok_or(FacadeError::Memory {
                    detail: "staging element byte count overflowed usize".to_owned(),
                })?;
        let mut owned_values: Vec<F> = Vec::new();
        owned_values
            .try_reserve_exact(staging_elements)
            .map_err(|_| FacadeError::Memory {
                detail: format!("failed to allocate staging buffer of {staging_bytes} bytes"),
            })?;
        owned_values.resize(staging_elements, F::zero());

        let schedule = schedule_chunks(&plan.workspace);
        let mut total_not0: i32 = 0;
        let mut total_transfer_bytes: usize = 0;
        let mut total_peak_workspace_bytes: usize = 0;

        for chunk in schedule.chunks() {
            // FND-06 / 25-02 chunk-staging contract (chunk-aware proven-sized output).
            //
            // The family kernels are MONOLITHIC whole-block writers: each launch
            // computes ALL components and ALL AO pairs of the shell tuple and scatters
            // them at ABSOLUTE output indices `[0, staging_elements)`. The executor
            // does NOT translate `chunk.work_unit_start` into a scatter offset. So
            // memory-limit chunking partitions the *workspace* (compute scratch,
            // `chunk.bytes`) — NOT the output buffer. Handing the kernel a
            // `chunk_len`-sized staging slice under-sizes the buffer (the absolute
            // `dst` overflows it); the now-removed `if dst < staging.len()` scatter
            // guards used to mask this with silent partial writes — the exact
            // anti-pattern FND-06 eliminates. Mirror of the fix in
            // `cintx-compat::eval_raw`. The OOM no-partial-write contract is preserved
            // upstream: `query_workspace` returns `MemoryLimitExceeded` (no staging
            // touched) when even one workspace chunk cannot fit the limit.
            //
            // chunk_staging is always Vec<f64> — the kernel dispatchers receive
            // &mut [f64] (frozen ExecutionIo interface) and reinterpret internally for
            // f32 precision. A Vec<f64> of `staging_elements` elements gives
            // `staging_elements * 2` f32 lanes; the f32 dispatcher bounds its writes to
            // `out_elems = staging.len()` pre-cast == `staging_elements`.
            let chunk_staging_bytes =
                staging_elements
                    .checked_mul(size_of::<f64>())
                    .ok_or(FacadeError::Memory {
                        detail: "chunk staging byte count overflowed usize".to_owned(),
                    })?;
            let mut chunk_staging: Vec<f64> = Vec::new();
            chunk_staging
                .try_reserve_exact(staging_elements)
                .map_err(|_| FacadeError::Memory {
                    detail: format!(
                        "failed to allocate chunk staging buffer of {chunk_staging_bytes} bytes"
                    ),
                })?;
            chunk_staging.resize(staging_elements, 0.0f64);

            let mut workspace = allocator
                .try_alloc(chunk.bytes, plan.workspace.alignment)
                .map_err(FacadeError::from)?;

            {
                let mut io =
                    ExecutionIo::new(chunk, &mut chunk_staging, &mut workspace, plan.dispatch)
                        .map_err(FacadeError::from)?;
                let chunk_stats = executor
                    .execute(&plan, &mut io)
                    .map_err(FacadeError::from)?;
                // Each chunk recomputes the same full block, so `not0` is the SAME
                // full-block nonzero count every chunk — take the representative
                // value (max), not a sum, to avoid N× over-counting under multi-chunk.
                total_not0 = total_not0.max(chunk_stats.not0.max(0));
                total_transfer_bytes = total_transfer_bytes
                    .saturating_add(io.transfer_bytes())
                    .saturating_add(chunk_stats.transfer_bytes);
                total_peak_workspace_bytes = total_peak_workspace_bytes.max(io.workspace().len());
            }
            allocator.release(workspace);

            // The kernel wrote the entire monolithic block into `chunk_staging`. Copy
            // the full block into the accumulator by reinterpreting the f64 buffer as
            // &[F] (f64: zero-cost identity cast; f32: the kernel wrote
            // `staging_elements` f32 values at indices 0..staging_elements in the f32
            // view). Each chunk recomputes the same full block (the kernel ignores
            // chunk boundaries for output), so the accumulator holds the complete,
            // correct output regardless of chunk_count.
            let chunk_as_f: &[F] = bytemuck::cast_slice::<f64, F>(&chunk_staging);
            let copy_len = chunk_as_f.len().min(staging_elements);
            owned_values[..copy_len].copy_from_slice(&chunk_as_f[..copy_len]);
        }

        if owned_values.len() != output_layout.staging_elements {
            return Err(FacadeError::Validation {
                detail: format!(
                    "owned output contract drift: expected staging_elements={} got={}",
                    output_layout.staging_elements,
                    owned_values.len()
                ),
            });
        }

        let bytes_written =
            owned_values
                .len()
                .checked_mul(size_of::<F>())
                .ok_or(FacadeError::Memory {
                    detail: "owned output byte size overflowed usize".to_owned(),
                })?;

        let chunk_count = schedule_chunks(&plan.workspace).len();
        let runtime_stats = ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: total_peak_workspace_bytes,
            chunk_count: chunk_count.max(plan.workspace.chunks.len()),
            planned_batches: plan
                .workspace
                .chunks
                .iter()
                .map(|c| c.work_unit_count)
                .sum(),
            transfer_bytes: total_transfer_bytes,
            not0: total_not0,
            fallback_reason: plan.workspace.fallback_reason,
        };

        let stats = EvaluationStats::from_runtime(&runtime_stats);

        Ok(TypedEvaluationOutput {
            tensor: IntegralTensor {
                extents: output_layout.extents,
                component_axis_leading: output_layout.component_axis_leading,
                complex_interleaved: output_layout.complex_interleaved,
                owned_values,
            },
            stats,
            workspace_bytes: runtime_stats.workspace_bytes,
            chunk_count: runtime_stats.chunk_count,
            bytes_written,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceChunk {
    pub index: usize,
    pub work_unit_start: usize,
    pub work_unit_count: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceExecutionToken {
    operator: OperatorId,
    representation: Representation,
    shell_count: usize,
    required_workspace_bytes: usize,
    memory_limit_bytes: Option<usize>,
    chunk_size_override: Option<usize>,
}

impl WorkspaceExecutionToken {
    pub fn operator(&self) -> OperatorId {
        self.operator
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn shell_count(&self) -> usize {
        self.shell_count
    }

    pub fn required_workspace_bytes(&self) -> usize {
        self.required_workspace_bytes
    }

    pub fn memory_limit_bytes(&self) -> Option<usize> {
        self.memory_limit_bytes
    }

    pub fn chunk_size_override(&self) -> Option<usize> {
        self.chunk_size_override
    }

    fn from_request(
        request: &SessionRequest<'_>,
        workspace: &RuntimeWorkspaceQuery,
    ) -> WorkspaceExecutionToken {
        WorkspaceExecutionToken {
            operator: request.operator,
            representation: request.representation,
            shell_count: request.shells.len(),
            required_workspace_bytes: workspace.required_bytes,
            memory_limit_bytes: request.options.memory_limit_bytes,
            chunk_size_override: request.options.chunk_size_override,
        }
    }

    fn ensure_matches(
        &self,
        request: &SessionRequest<'_>,
        runtime_workspace: &RuntimeWorkspaceQuery,
    ) -> Result<(), FacadeError> {
        if self.operator != request.operator
            || self.representation != request.representation
            || self.shell_count != request.shells.len()
            || self.required_workspace_bytes != runtime_workspace.required_bytes
            || self.memory_limit_bytes != request.options.memory_limit_bytes
            || self.chunk_size_override != request.options.chunk_size_override
        {
            return Err(FacadeError::Validation {
                detail: "query/evaluate contract drift detected before execution".to_owned(),
            });
        }

        if !runtime_workspace.planning_matches(&request.options) {
            return Err(FacadeError::Validation {
                detail:
                    "query/evaluate contract drift detected: planning_matches=false for options"
                        .to_owned(),
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspacePlan {
    pub bytes: usize,
    pub required_bytes: usize,
    pub chunk_count: usize,
    pub work_units: usize,
    pub fallback_reason: Option<&'static str>,
    pub chunks: Vec<WorkspaceChunk>,
    pub memory_limit_bytes: Option<usize>,
    pub chunk_size_override: Option<usize>,
    pub execution_token: WorkspaceExecutionToken,
}

impl WorkspacePlan {
    fn from_runtime(request: &SessionRequest<'_>, runtime: &RuntimeWorkspaceQuery) -> Self {
        Self {
            bytes: runtime.bytes,
            required_bytes: runtime.required_bytes,
            chunk_count: runtime.chunk_count,
            work_units: runtime.work_units,
            fallback_reason: runtime.fallback_reason,
            chunks: runtime
                .chunks
                .iter()
                .map(|chunk| WorkspaceChunk {
                    index: chunk.index,
                    work_unit_start: chunk.work_unit_start,
                    work_unit_count: chunk.work_unit_count,
                    bytes: chunk.bytes,
                })
                .collect(),
            memory_limit_bytes: runtime.memory_limit_bytes,
            chunk_size_override: runtime.chunk_size_override,
            execution_token: WorkspaceExecutionToken::from_request(request, runtime),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluationStats {
    pub workspace_bytes: usize,
    pub required_workspace_bytes: usize,
    pub peak_workspace_bytes: usize,
    pub chunk_count: usize,
    pub planned_batches: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
    pub fallback_reason: Option<&'static str>,
}

impl EvaluationStats {
    fn from_runtime(stats: &ExecutionStats) -> Self {
        Self {
            workspace_bytes: stats.workspace_bytes,
            required_workspace_bytes: stats.required_workspace_bytes,
            peak_workspace_bytes: stats.peak_workspace_bytes,
            chunk_count: stats.chunk_count,
            planned_batches: stats.planned_batches,
            transfer_bytes: stats.transfer_bytes,
            not0: stats.not0,
            fallback_reason: stats.fallback_reason,
        }
    }
}

/// Owned integral tensor returned by `SessionQuery::evaluate`.
///
/// # Memory layout
///
/// `owned_values` is a dense `Vec<F>` (default `f64`) storing `extents.iter().product()` real
/// values (or 2x that for `Spinor` outputs with `complex_interleaved == true`,
/// where real and imaginary parts alternate in the innermost stride).
///
/// For complex outputs (`complex_interleaved == true`, e.g. Spinor), call `complex_values()`
/// to get the typed `Vec<Complex<F>>` view (D-04 / SC-2); `owned_values` remains the
/// underlying interleaved storage.
///
/// The type parameter `F` is the output float precision (`f64` or `f32`). The default
/// `F = f64` keeps every existing call site compiling unchanged (D-12).
///
/// **AO axis layout** — `extents` lists AO-axis sizes in **shell-tuple order**:
/// `extents[0] = ao_per_shell(shells[0])`, `extents[1] = ao_per_shell(shells[1])`,
/// etc. The per-kernel index ordering inside `owned_values` matches libcint's
/// memory layout for that family:
///
/// - **Arity >= 3** (`int2e_*`, `int3c1e_*`, `int3c2e_*`, `int4c1e_*`): **F-order**
///   (Fortran / column-major) — `extents[0]` is the fastest-varying axis.
///   Byte-identical to vendor libcint output without transposition (verified by
///   the Phase 18 oracle parity sweep in `crates/cintx-oracle/tests/safe_api_arity{3,4}_parity.rs`).
/// - **Arity 2** (`int1e_*`, `int2c2e_*`): row-major within each shell-pair
///   block — `extents[0]` is the slowest-varying axis. The arity-2 oracle parity
///   tests apply the column-major-to-row-major conversion to vendor output before
///   comparison (see `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:280-292`).
///
/// When `component_axis_leading == true` (the planner default), an optional
/// component axis (e.g., for IP/derivative operators) is the slowest-varying
/// axis — placed beyond `extents.len()` shell-tuple axes.
///
/// The arity-aware layout is verified implicitly by the oracle parity sweep
/// (`crates/cintx-oracle/tests/safe_api_arity{2,3,4}_parity.rs`). If the layout
/// silently drifts, the first parity test fails.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor<F = f64> {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<F>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedEvaluationOutput<F = f64> {
    pub tensor: IntegralTensor<F>,
    pub stats: EvaluationStats,
    pub workspace_bytes: usize,
    pub chunk_count: usize,
    pub bytes_written: usize,
}

impl<F: CintFloat> IntegralTensor<F> {
    /// Complex view of the output for spinor/complex operators (D-04 / SC-2).
    ///
    /// Returns `Some(Vec<Complex<F>>)` when `complex_interleaved == true` (Spinor
    /// outputs): the contiguous interleaved `[re, im, re, im, ...]` `owned_values`
    /// buffer is reinterpreted element-for-element into `Complex<F>` (num_complex's
    /// `Complex<F>` is `#[repr(C)] { re, im }`, contiguous — this is a typed
    /// reinterpretation, not a data reshuffle). Returns `None` for real-valued
    /// outputs, where callers consume `owned_values: Vec<F>` directly.
    ///
    /// Migration note: callers that previously consumed the interleaved `Vec<F>` for
    /// complex outputs (manually pairing `owned_values[2i]`/`owned_values[2i+1]`) should
    /// now call `tensor.complex_values()` for the typed `Vec<Complex<F>>` view.
    pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>> {
        if !self.complex_interleaved {
            return None;
        }
        // owned_values.len() is even for complex_interleaved (re/im pairs).
        debug_assert_eq!(self.owned_values.len() % 2, 0);
        Some(
            self.owned_values
                .chunks_exact(2)
                .map(|pair| num_complex::Complex::new(pair[0], pair[1]))
                .collect(),
        )
    }
}

impl<F: CintFloat> TypedEvaluationOutput<F> {
    /// Convenience: complex view of the tensor (see `IntegralTensor::complex_values`).
    pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>> {
        self.tensor.complex_values()
    }
}

/// Explicit fallback used when unstable source requests are attempted without feature support.
pub fn unsupported_unstable_request(symbol: &str) -> FacadeError {
    FacadeError::UnsupportedApi {
        requested: format!(
            "unstable source symbol `{symbol}` requires feature `unstable-source-api`"
        ),
    }
}

#[cfg(feature = "unstable-source-api")]
pub mod unstable {
    //! Source-only namespace that remains opt-in until manifest/oracle release gates and
    //! explicit maintainer approval promote entries into the stable facade.

    /// Marker payload for source-only API entries that are not part of the stable facade namespace.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SourceApiToken {
        pub family: &'static str,
        pub symbol: &'static str,
    }

    impl SourceApiToken {
        pub const fn new(family: &'static str, symbol: &'static str) -> Self {
            Self { family, symbol }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BatchRequest, EvaluationContext, EvaluationStats, IntegralTensor, Session, SessionRequest,
        TypedEvaluationOutput, ss_batch_inputs, unsupported_unstable_request,
    };
    use crate::error::{FacadeError, FacadeErrorKind};
    #[cfg(feature = "with-f12")]
    use cintx_compat::raw::enforce_safe_facade_policy_gate;
    use cintx_core::ecp::{EcpChannel, EcpShell};
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_runtime::ExecutionOptions;
    #[cfg(feature = "with-f12")]
    use cintx_runtime::{ExecutionPlan, query_workspace as runtime_query_workspace};
    use std::sync::Arc;

    #[cfg(feature = "with-4c1e")]
    const INT4C1E_CART_OPERATOR_ID: u32 = 24;
    // Phase 19 Plan 01 inserted four `int1e_ecp_*` rows at OperatorIds 26..=29,
    // ahead of the F12 / source-only blocks; every operator id at or after 26
    // shifts by +4. The values below are derived from
    // crates/cintx-ops/src/generated/api_manifest.rs::OPERATOR_DESCRIPTORS
    // post-regeneration (verified manually: int2e_stg_sph is at position 106,
    // int2e_ipip1_sph is at position 116).
    #[cfg(feature = "with-f12")]
    const INT2E_STG_SPH_OPERATOR_ID: u32 = 106;
    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis_with_shells(
        rep: Representation,
        shell_l_values: &[u8],
    ) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let mut shells = Vec::new();
        for (idx, shell_l) in shell_l_values.iter().copied().enumerate() {
            let exponent = 1.0 - (idx as f64 * 0.05);
            let shell = Arc::new(
                Shell::try_new(
                    0,
                    shell_l,
                    1,
                    1,
                    0,
                    rep,
                    arc_f64(&[exponent]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            );
            shells.push(shell);
        }

        let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
        let shell_tuple = ShellTuple::try_from_iter(shells).unwrap();
        (basis, shell_tuple)
    }

    fn sample_basis(rep: Representation) -> (BasisSet, ShellTuple) {
        sample_basis_with_shells(rep, &[1, 1])
    }

    #[test]
    fn query_workspace_returns_structured_contract_metadata() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let workspace = query.workspace();

        assert!(workspace.bytes > 0);
        assert_eq!(workspace.chunk_count, workspace.chunks.len());
        assert_eq!(workspace.execution_token.operator(), OperatorId::new(0));
        assert_eq!(
            workspace.execution_token.representation(),
            Representation::Cart
        );
        assert_eq!(workspace.execution_token.shell_count(), 2);
        assert_eq!(
            workspace.execution_token.required_workspace_bytes(),
            workspace.required_bytes
        );
    }

    #[test]
    fn evaluate_returns_deterministic_nonzero_real_values() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0), // int1e_ovlp_cart
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        // Capture expected workspace metadata before consuming the first query.
        let query1 = request
            .clone()
            .query_workspace()
            .expect("query should succeed");
        let expected_workspace_bytes = query1.workspace().bytes;
        let expected_chunk_count = query1.workspace().chunk_count;

        // First evaluation.
        let output1 = query1.evaluate().expect("safe evaluate should succeed");

        // Second independent evaluation from the same request — idempotency check.
        let query2 = request
            .query_workspace()
            .expect("query should succeed (2nd)");
        let output2 = query2
            .evaluate()
            .expect("safe evaluate should succeed (2nd)");

        // (1) Idempotency: real, deterministic kernel must return identical values across runs.
        assert_eq!(
            output1.tensor.owned_values, output2.tensor.owned_values,
            "evaluate must be deterministic across repeated calls"
        );

        // (2) Nonzero: a zero-fill regression must fail this test. For two valid GTO shells the
        //     overlap matrix must contain at least one nonzero element (self-overlap of normalized
        //     GTOs is nonzero on the diagonal).
        let nonzero_count = output1
            .tensor
            .owned_values
            .iter()
            .filter(|&&v| v.abs() > 1e-18)
            .count();
        assert!(
            nonzero_count > 0,
            "evaluate must produce at least one nonzero element; got all-zero owned_values \
             (regression to zero-fill stub?)"
        );

        // (3) Existing extent / byte-count / stats invariants (preserved from prior test).
        assert!(!output1.tensor.owned_values.is_empty());
        assert_eq!(
            output1.tensor.owned_values.len(),
            output1.tensor.extents.iter().product::<usize>(),
        );
        assert_eq!(output1.workspace_bytes, expected_workspace_bytes);
        assert_eq!(output1.chunk_count, expected_chunk_count);
        assert_eq!(
            output1.bytes_written,
            output1.tensor.owned_values.len() * std::mem::size_of::<f64>(),
        );
        assert!(output1.stats.transfer_bytes > 0);
    }

    #[test]
    fn query_evaluate_contract_drift_is_detected_before_execution() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions {
                memory_limit_bytes: Some(192),
                ..ExecutionOptions::default()
            },
        );

        let mut query = request.query_workspace().expect("query should succeed");
        query.request.options.memory_limit_bytes = Some(256);

        let err = query.evaluate().unwrap_err();
        assert!(matches!(err, FacadeError::Validation { .. }));
        assert!(err.to_string().contains("contract drift"));
    }

    #[cfg(feature = "with-f12")]
    #[test] // unsupported
    fn compat_policy_gate_reports_with_f12_sph_envelope_reason_in_safe_module() {
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[1, 1, 1, 1]);
        let request = SessionRequest::new(
            OperatorId::new(INT2E_STG_SPH_OPERATOR_ID),
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let runtime_workspace = runtime_query_workspace(
            request.operator(),
            request.representation(),
            request.basis(),
            request.shells().clone(),
            request.options(),
        )
        .expect("with-f12 query should succeed");
        let plan = ExecutionPlan::new(
            request.operator(),
            request.representation(),
            request.basis(),
            request.shells().clone(),
            &runtime_workspace,
        )
        .expect("with-f12 execution plan should build");

        let err = enforce_safe_facade_policy_gate(
            plan.descriptor,
            Representation::Cart,
            request.shells(),
            &plan.output_layout.extents,
        )
        .map_err(FacadeError::from)
        .unwrap_err();
        assert!(matches!(
            err,
            FacadeError::UnsupportedApi { requested }
                if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test] // unsupported validated4c1e
    fn evaluate_rejects_out_of_envelope_validated4c1e_requests() {
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[5, 1, 1, 1]);
        let request = SessionRequest::new(
            OperatorId::new(INT4C1E_CART_OPERATOR_ID),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();
        assert!(matches!(
            err,
            FacadeError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test] // source unsupported
    fn evaluate_rejects_source_only_symbols_via_compat_policy_gate() {
        // int2e_ipip1_sph was promoted to stable in Phase 25 HESS-02 (25-04, D-07),
        // so it no longer exercises the source-only gate. Resolve a still-source-only
        // symbol BY NAME so this test survives the OperatorId reordering that adding
        // manifest rows causes — do NOT hardcode a numeric id (the old `= 116` constant
        // silently came to point at int1e_r2_origi_sph after the Phase-25 rows landed).
        let operator = cintx_ops::resolver::Resolver::descriptor_by_symbol("int1e_r2_origi_sph")
            .expect("source-only symbol must exist in manifest")
            .id;
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[1, 1]);
        let request = SessionRequest::new(
            operator,
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();
        match err {
            FacadeError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("source-only symbol")
                        && requested.contains("unstable-source-api"),
                    "unexpected unsupported reason: {requested}"
                );
            }
            other => panic!("expected UnsupportedApi error, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_unstable_requests_map_to_unsupported_api() {
        let err = unsupported_unstable_request("int2e_ipip1_sph");
        assert!(matches!(err, FacadeError::UnsupportedApi { .. }));
    }

    #[test]
    fn aosym_error_path_rejects_non_s1_with_typed_error() {
        use cintx_core::AoSymmetry;
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

        for non_s1 in [
            AoSymmetry::S2ij,
            AoSymmetry::S2kl,
            AoSymmetry::S4,
            AoSymmetry::S8,
        ] {
            let options = ExecutionOptions {
                aosym: Some(non_s1),
                ..Default::default()
            };
            let request = SessionRequest::new(
                OperatorId::new(0),
                Representation::Cart,
                &basis,
                shells.clone(),
                options,
            );
            let err = request
                .query_workspace()
                .expect_err("non-S1 aosym must return UnsupportedAoSymmetry");
            match err {
                FacadeError::UnsupportedAoSymmetry { requested } => {
                    assert_eq!(
                        requested,
                        non_s1.to_string(),
                        "requested field must carry the lowercase pyscf form"
                    );
                }
                other => {
                    panic!("expected UnsupportedAoSymmetry for aosym={non_s1:?}, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn aosym_none_and_s1_both_succeed_through_query_workspace() {
        use cintx_core::AoSymmetry;
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

        for aosym in [None, Some(AoSymmetry::S1)] {
            let options = ExecutionOptions {
                aosym,
                ..Default::default()
            };
            let request = SessionRequest::new(
                OperatorId::new(0),
                Representation::Cart,
                &basis,
                shells.clone(),
                options,
            );
            request
                .query_workspace()
                .unwrap_or_else(|e| panic!("aosym={aosym:?} must succeed; got {e:?}"));
        }
    }

    // ----------------------------------------------------------------------
    // Phase 19 D-06: FacadeError::MissingEcpBasis variant + query_workspace
    // preflight tests.
    // ----------------------------------------------------------------------

    fn sample_basis_with_ecp(
        rep: Representation,
        shell_l_values: &[u8],
        ecp_count: usize,
    ) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let mut shells = Vec::new();
        for (idx, shell_l) in shell_l_values.iter().copied().enumerate() {
            let exponent = 1.0 - (idx as f64 * 0.05);
            let shell = Arc::new(
                Shell::try_new(
                    0,
                    shell_l,
                    1,
                    1,
                    0,
                    rep,
                    arc_f64(&[exponent]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            );
            shells.push(shell);
        }

        let mut ecp_shells: Vec<Arc<EcpShell>> = Vec::new();
        for _ in 0..ecp_count {
            ecp_shells.push(Arc::new(
                EcpShell::try_new(
                    0,
                    EcpChannel::Local,
                    0,
                    1,
                    1,
                    0,
                    arc_f64(&[0.5]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            ));
        }
        let ecp_arc = Arc::from(ecp_shells.into_boxed_slice());

        let basis = BasisSet::try_new_with_ecp(
            atoms,
            Arc::from(shells.clone().into_boxed_slice()),
            ecp_arc,
        )
        .unwrap();
        let shell_tuple = ShellTuple::try_from_iter(shells).unwrap();
        (basis, shell_tuple)
    }

    #[test]
    fn facade_error_missing_ecp_basis_carries_kind_and_operator() {
        // Test 1+2 (variant exists, kind() arm wired).
        let err = FacadeError::MissingEcpBasis {
            operator: "int1e_ecp_sph".to_owned(),
        };
        assert_eq!(err.kind(), FacadeErrorKind::MissingEcpBasis);
        match &err {
            FacadeError::MissingEcpBasis { operator } => {
                assert_eq!(operator, "int1e_ecp_sph");
            }
            other => panic!("expected MissingEcpBasis variant, got {other:?}"),
        }
    }

    #[test]
    fn facade_error_missing_ecp_basis_display_message_matches_pattern() {
        // Test 3: Display message matches the documented contract.
        let err = FacadeError::MissingEcpBasis {
            operator: "int1e_ecp_sph".to_owned(),
        };
        let rendered = format!("{err}");
        assert!(
            rendered.contains("operator 'int1e_ecp_sph'")
                && rendered.contains("requires ECP basis")
                && rendered.contains("BasisSet::ecp_shells()")
                && rendered.contains("is empty"),
            "unexpected Display rendering: {rendered}",
        );
    }

    #[test]
    fn query_workspace_returns_missing_ecp_basis_for_ecp_op_without_ecp_shells() {
        // Test 4: ECP operator + ECP-less basis → MissingEcpBasis preflight
        // fires before any runtime_query_workspace work.
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[0, 0]);
        assert!(basis.ecp_shells().is_empty(), "fixture must have no ECP");
        let request = SessionRequest::new(
            OperatorId::INT1E_ECP_SPH,
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let err = request
            .query_workspace()
            .expect_err("ECP op without ECP basis must fail preflight");
        match err {
            FacadeError::MissingEcpBasis { operator } => {
                assert_eq!(operator, "int1e_ecp_sph");
            }
            other => panic!("expected MissingEcpBasis, got {other:?}"),
        }
    }

    #[test]
    fn query_workspace_does_not_return_missing_ecp_basis_for_non_ecp_operators() {
        // Test 5: non-ECP operator must not trip the MissingEcpBasis gate
        // regardless of whether ecp_shells is empty.
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[0, 0]);
        assert!(basis.ecp_shells().is_empty());
        let request = SessionRequest::new(
            // int1e_ovlp_sph — not an ECP operator
            OperatorId::new(1),
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        // query_workspace may succeed or return a non-MissingEcpBasis error
        // (e.g. UnsupportedApi for some configurations), but it must NEVER
        // return MissingEcpBasis for a non-ECP operator.
        match request.query_workspace() {
            Ok(_) => {}
            Err(FacadeError::MissingEcpBasis { .. }) => {
                panic!("non-ECP operator must not return MissingEcpBasis");
            }
            Err(_) => {}
        }
    }

    #[test]
    fn query_workspace_passes_through_for_ecp_op_with_ecp_shells_attached() {
        // Test 6: ECP operator + ECP basis attached → preflight does not
        // block. The call may fail later (planner / executor / no kernel
        // wired yet in Plan 03), but the preflight itself must not return
        // MissingEcpBasis.
        let (basis, shells) = sample_basis_with_ecp(Representation::Spheric, &[0, 0], 1);
        assert_eq!(basis.ecp_shells().len(), 1);
        let request = SessionRequest::new(
            OperatorId::INT1E_ECP_SPH,
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        match request.query_workspace() {
            Ok(_) => {}
            Err(FacadeError::MissingEcpBasis { .. }) => {
                panic!("ECP op with ecp_shells attached must NOT return MissingEcpBasis");
            }
            Err(_) => {
                // Plan 04 wires the kernel; until then a downstream error
                // is acceptable. The point of this test is purely that the
                // ECP preflight does not fire.
            }
        }
    }

    // ------------------------------------------------------------------
    // Plan 20-07 Task 1: generic output structs IntegralTensor<F> /
    // TypedEvaluationOutput<F> with f64 defaults.
    // ------------------------------------------------------------------

    #[test]
    fn integral_tensor_default_type_param_is_f64() {
        // Unparameterized IntegralTensor resolves to IntegralTensor<f64>.
        // Compile-time test: this must type-check without a turbofish.
        let t: IntegralTensor = IntegralTensor::default();
        // owned_values must be Vec<f64>
        let _: Vec<f64> = t.owned_values;
    }

    // ------------------------------------------------------------------
    // Plan 20-07 Task 2: generic evaluate::<F: CintFloat>() and f64 shim.
    // ------------------------------------------------------------------

    #[test]
    fn cintfloat_precision_const_f64_is_f64() {
        use cintx_core::{CintFloat, PrecisionKind};
        assert_eq!(<f64 as CintFloat>::PRECISION, PrecisionKind::F64);
    }

    #[test]
    fn cintfloat_precision_const_f32_is_f32() {
        use cintx_core::{CintFloat, PrecisionKind};
        assert_eq!(<f32 as CintFloat>::PRECISION, PrecisionKind::F32);
    }

    #[test]
    fn evaluate_generic_f32_returns_vec_f32_with_nonzero_element() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0), // int1e_ovlp_cart
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let query = request.query_workspace().expect("query should succeed");
        let output: TypedEvaluationOutput<f32> = query
            .evaluate_generic::<f32>()
            .expect("evaluate::<f32> should succeed");
        // Must return Vec<f32>
        let _: Vec<f32> = output.tensor.owned_values.clone();
        // Must have at least one nonzero element
        let nonzero_count = output
            .tensor
            .owned_values
            .iter()
            .filter(|&&v| v.abs() > 1e-9_f32)
            .count();
        assert!(
            nonzero_count > 0,
            "evaluate::<f32> must produce at least one nonzero element"
        );
    }

    #[test]
    fn evaluate_unparameterized_delegates_to_f64_path() {
        // evaluate() (unparameterized) must return TypedEvaluationOutput<f64> byte-identically.
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            ExecutionOptions::default(),
        );
        let q1 = request.clone().query_workspace().expect("q1");
        let out1 = q1.evaluate().expect("evaluate f64 unparameterized");
        let q2 = request.query_workspace().expect("q2");
        let out2 = q2
            .evaluate_generic::<f64>()
            .expect("evaluate_generic::<f64>");
        // Must be byte-identical (same f64 values)
        assert_eq!(
            out1.tensor.owned_values, out2.tensor.owned_values,
            "evaluate() and evaluate_generic::<f64>() must be byte-identical"
        );
    }

    #[test]
    fn typed_evaluation_output_default_type_param_is_f64() {
        // Unparameterized TypedEvaluationOutput resolves to TypedEvaluationOutput<f64>.
        let out: TypedEvaluationOutput = TypedEvaluationOutput::default();
        let _: Vec<f64> = out.tensor.owned_values;
    }

    #[test]
    fn integral_tensor_f32_is_constructible() {
        // TypedEvaluationOutput<f32> and IntegralTensor<f32> must be constructible.
        let t: IntegralTensor<f32> = IntegralTensor {
            extents: vec![2, 2],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        };
        assert_eq!(t.owned_values.len(), 4);
        let out: TypedEvaluationOutput<f32> = TypedEvaluationOutput {
            tensor: t,
            stats: EvaluationStats::default(),
            workspace_bytes: 0,
            chunk_count: 0,
            bytes_written: 0,
        };
        assert_eq!(out.tensor.owned_values[0], 1.0_f32);
    }

    // --- D-04 / ROADMAP SC-2 / Gap 1: Complex<F> typed view tests ---

    #[test]
    fn complex_values_returns_some_for_complex_interleaved_f64() {
        // Spinor output (complex_interleaved == true): complex_values() returns Some.
        // Buffer: [1.0, 2.0, 3.0, 4.0] -> [Complex(1,2), Complex(3,4)]
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64],
        };
        let cv = t.complex_values();
        assert!(
            cv.is_some(),
            "complex_values() must return Some when complex_interleaved == true"
        );
        let v = cv.unwrap();
        assert_eq!(v.len(), 2, "len must be owned_values.len() / 2");
        assert_eq!(v[0], num_complex::Complex::new(1.0_f64, 2.0_f64));
        assert_eq!(v[1], num_complex::Complex::new(3.0_f64, 4.0_f64));
    }

    #[test]
    fn complex_values_returns_none_for_real_output() {
        // Real output (complex_interleaved == false): complex_values() returns None.
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![1.0_f64, 2.0_f64],
        };
        assert!(
            t.complex_values().is_none(),
            "complex_values() must return None when complex_interleaved == false"
        );
    }

    #[test]
    fn complex_values_f32_typed() {
        // IntegralTensor<f32> complex view yields Complex<f32>.
        let t: IntegralTensor<f32> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        };
        let cv = t.complex_values();
        assert!(cv.is_some());
        let v: Vec<num_complex::Complex<f32>> = cv.unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], num_complex::Complex::new(1.0_f32, 2.0_f32));
        assert_eq!(v[1], num_complex::Complex::new(3.0_f32, 4.0_f32));
    }

    #[test]
    fn typed_evaluation_output_complex_values_delegates_to_tensor() {
        // TypedEvaluationOutput::complex_values() delegates to tensor.complex_values().
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![1],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![5.0_f64, 6.0_f64],
        };
        let out = TypedEvaluationOutput {
            tensor: t,
            stats: EvaluationStats::default(),
            workspace_bytes: 0,
            chunk_count: 0,
            bytes_written: 0,
        };
        let cv = out.complex_values();
        assert!(cv.is_some());
        let v = cv.unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], num_complex::Complex::new(5.0_f64, 6.0_f64));
    }

    #[test]
    fn owned_values_field_unchanged_by_complex_view() {
        // The owned_values: Vec<F> field is accessible and byte-identical (PREC-04 / SemVer).
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64],
        };
        // Field must still be accessible as Vec<f64> directly.
        let raw: &Vec<f64> = &t.owned_values;
        assert_eq!(raw, &vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]);
    }

    /// Task 2: PREC-02 end-to-end smoke — a real spinor evaluate() exposes complex_values() == Some.
    ///
    /// This test drives `SessionRequest::evaluate()` with `Representation::Spinor` (int1e_ovlp_spinor,
    /// OperatorId 2) and asserts:
    /// 1. `output.complex_values().is_some()` — PREC-02/D-04/SC-2 literally TRUE.
    /// 2. `complex_values().unwrap().len() == output.tensor.owned_values.len() / 2` — correct pairing.
    /// 3. `owned_values` is unchanged (PREC-04 regression guard on the accessor path).
    #[test]
    fn spinor_evaluate_exposes_complex_values_some_prec02() {
        // int1e_ovlp_spinor = OperatorId 2; two s-shells (l=0) with Spinor representation.
        let (basis, shells) = sample_basis(Representation::Spinor);
        let request = SessionRequest::new(
            OperatorId::new(2), // int1e_ovlp_spinor
            Representation::Spinor,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let query = request
            .query_workspace()
            .expect("spinor query_workspace must succeed");
        let output = query.evaluate().expect("spinor evaluate must succeed");

        // PREC-02: complex_values() must return Some for Spinor output.
        assert!(
            output.complex_values().is_some(),
            "PREC-02 GAP-1: complex_values() must return Some for Spinor evaluate output (complex_interleaved == true)"
        );
        // Correct pairing: len == owned_values.len() / 2.
        let cv = output.complex_values().unwrap();
        assert_eq!(
            cv.len(),
            output.tensor.owned_values.len() / 2,
            "complex_values() length must be owned_values.len() / 2"
        );
        // PREC-04: owned_values still accessible and unchanged.
        assert!(
            !output.tensor.owned_values.is_empty(),
            "owned_values must be non-empty (PREC-04 regression guard)"
        );
        // Spinor overlap must produce at least one nonzero complex element.
        let nonzero_count = cv.iter().filter(|c| c.norm_sqr() > 1e-36).count();
        assert!(
            nonzero_count > 0,
            "spinor evaluate must produce at least one nonzero Complex<f64> element"
        );
    }

    // Phase 22 FND-01 gap closure: the safe-API path must finiteness-validate the gauge
    // origin, matching the raw-path guard in cintx-compat raw.rs. Before this fix the
    // builder doc-contract ("NaN/inf rejected") held only on the raw path; a
    // `.with_common_origin([NaN, ..])` caller silently threaded garbage into the plan.
    #[test]
    fn evaluate_rejects_non_finite_common_origin_on_safe_api_path() {
        use crate::builder::SessionBuilder;

        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionBuilder::new(OperatorId::new(0), Representation::Cart, &basis, shells)
            .with_common_origin([f64::NAN, 0.0, 0.0])
            .build();

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();

        assert_eq!(
            err.kind(),
            FacadeErrorKind::Validation,
            "non-finite gauge origin must surface as a Validation error on the safe path"
        );
        assert!(
            err.to_string().contains("PTR_COMMON_ORIG"),
            "error must name the PTR_COMMON_ORIG param; got: {err}"
        );
    }

    // Companion guard: a finite gauge origin on the safe-API path is accepted and
    // round-trips into the plan (no false rejection from the new validate call).
    #[test]
    fn evaluate_accepts_finite_common_origin_on_safe_api_path() {
        use crate::builder::SessionBuilder;

        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionBuilder::new(OperatorId::new(0), Representation::Cart, &basis, shells)
            .with_common_origin([0.5, -0.3, 0.8])
            .build();

        let query = request.query_workspace().expect("query should succeed");
        query
            .evaluate()
            .expect("finite gauge origin must be accepted on the safe path");
    }

    #[test]
    fn batch_evaluation_matches_scalar_order_and_offsets() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let first = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            ExecutionOptions::default(),
        );
        let scalar = first.clone().query_workspace().unwrap().evaluate().unwrap();
        let second = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let batch = BatchRequest::new([first, second])
            .max_items_per_chunk(1)
            .evaluate_batch()
            .unwrap();

        assert_eq!(batch.outputs.len(), 2);
        assert_eq!(
            batch.outputs[0].tensor.owned_values,
            scalar.tensor.owned_values
        );
        assert_eq!(batch.plan.items.len(), 2);
        assert_eq!(batch.plan.items[0].output_offset, 0);
        assert_eq!(
            batch.plan.items[1].output_offset,
            batch.plan.items[0].output_elements
        );
        assert_eq!(batch.plan.chunks.len(), 2);
        assert_eq!(batch.stats.items_planned, 2);
        assert_eq!(batch.stats.items_executed, 2);
        assert_eq!(batch.stats.kernel_launch_count, 2);
        assert_eq!(batch.stats.readback_count, 2);
    }

    #[test]
    fn batch_ss_pilots_match_compatibility_and_use_one_cached_client() {
        let atoms = Arc::from(
            vec![
                Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [0.0, 0.0, 1.4], NuclearModel::Point, None, None).unwrap(),
            ]
            .into_boxed_slice(),
        );
        let shell_i = Arc::new(
            Shell::try_new(
                0,
                0,
                3,
                1,
                0,
                Representation::Cart,
                arc_f64(&[1.24, 0.58, 0.17]),
                arc_f64(&[0.41, -0.26, 0.09]),
            )
            .unwrap(),
        );
        let shell_j = Arc::new(
            Shell::try_new(
                1,
                0,
                2,
                1,
                0,
                Representation::Cart,
                arc_f64(&[0.78, 0.31]),
                arc_f64(&[0.72, 0.18]),
            )
            .unwrap(),
        );
        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_i.clone(), shell_j.clone()].into_boxed_slice()),
        )
        .unwrap();
        let shells = ShellTuple::try_from_iter([shell_i, shell_j]).unwrap();
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            ExecutionOptions::default(),
        );
        let scalar = request
            .clone()
            .query_workspace()
            .unwrap()
            .evaluate()
            .unwrap();
        let context = EvaluationContext::new();
        let batch = BatchRequest::new(vec![request; 65])
            .max_items_per_chunk(13)
            .evaluate_batch_in(&context)
            .unwrap();

        assert_eq!(batch.outputs.len(), 65);
        assert_eq!(batch.plan.chunks.len(), 5);
        assert_eq!(batch.stats.items_planned, 65);
        assert_eq!(batch.stats.items_executed, 65);
        assert_eq!(batch.stats.bucket_count, 1);
        assert_eq!(batch.stats.kernel_launch_count, 5);
        assert_eq!(batch.stats.readback_count, 1);
        // Every item carries padded 3/2 primitive descriptor rows: four f64
        // primitive tables, two f64 centers, two u32 counts, and one f64 result.
        assert_eq!(batch.stats.transfer_bytes, 65 * 144);
        assert_eq!(
            batch
                .plan
                .chunks
                .iter()
                .flat_map(|chunk| batch.plan.chunk_items(chunk))
                .copied()
                .collect::<Vec<_>>(),
            (0..65).collect::<Vec<_>>(),
            "the pilot must submit each planned input exactly once"
        );
        for output in &batch.outputs {
            assert_eq!(output.tensor.extents, vec![1, 1]);
            assert_eq!(output.tensor.owned_values.len(), 1);
            assert!(
                (output.tensor.owned_values[0] - scalar.tensor.owned_values[0]).abs()
                    <= 1e-14 * scalar.tensor.owned_values[0].abs().max(1.0),
                "batched s-s overlap must remain within the f64 parity tolerance"
            );
            assert_eq!(output.stats.planned_batches, 1);
            assert_eq!(output.stats.transfer_bytes, 144);
            assert_eq!(output.workspace_bytes, 0);
        }

        let kinetic_request = SessionRequest::new(
            OperatorId::new(3),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let kinetic_scalar = kinetic_request
            .clone()
            .query_workspace()
            .unwrap()
            .evaluate()
            .unwrap();
        let kinetic_batch = BatchRequest::new(vec![kinetic_request; 65])
            .max_items_per_chunk(13)
            .evaluate_batch_in(&context)
            .unwrap();

        assert_eq!(kinetic_batch.stats.kernel_launch_count, 5);
        assert_eq!(kinetic_batch.stats.readback_count, 1);
        for output in &kinetic_batch.outputs {
            assert!(
                (output.tensor.owned_values[0] - kinetic_scalar.tensor.owned_values[0]).abs()
                    <= 1e-14 * kinetic_scalar.tensor.owned_values[0].abs().max(1.0),
                "batched s-s kinetic must remain within the f64 parity tolerance"
            );
        }
        assert_eq!(context.stats().backend_cache_entries, 1);
    }

    #[test]
    fn batch_primitive_eri_ssss_matches_scalar_compatibility_including_coincident_centers() {
        let atoms = Arc::from(
            vec![
                Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [0.3, -0.4, 0.7], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [-0.2, 0.8, -0.1], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [1.1, -0.6, 0.2], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [-0.7, 0.5, 0.4], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [0.2, 0.1, -1.0], NuclearModel::Point, None, None).unwrap(),
                Atom::try_new(1, [0.9, 0.3, 0.6], NuclearModel::Point, None, None).unwrap(),
            ]
            .into_boxed_slice(),
        );
        let shells: Vec<_> = [0u32, 1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .enumerate()
            .map(|(index, atom_index)| {
                Arc::new(
                    Shell::try_new(
                        atom_index,
                        0,
                        1,
                        1,
                        0,
                        Representation::Cart,
                        arc_f64(&[0.31 + index as f64 * 0.17]),
                        arc_f64(&[if index % 2 == 0 { 0.73 } else { -0.41 }]),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
        let make_request = |indices: [usize; 4]| {
            SessionRequest::new(
                cintx_ops::resolver::Resolver::descriptor_by_symbol("int2e_cart")
                    .unwrap()
                    .id,
                Representation::Cart,
                &basis,
                ShellTuple::try_from_iter(indices.map(|index| shells[index].clone())).unwrap(),
                ExecutionOptions::default(),
            )
        };
        let first = make_request([0, 1, 2, 3]); // A == B exercises T=0 on one pair.
        let second = make_request([4, 5, 6, 7]);
        let third = make_request([0, 3, 5, 6]);
        let expected = [first.clone(), second.clone(), third.clone()].map(|request| {
            request
                .query_workspace()
                .unwrap()
                .evaluate()
                .unwrap()
                .tensor
                .owned_values[0]
        });
        let context = EvaluationContext::new();
        let batch = BatchRequest::new([first, second, third])
            .max_items_per_chunk(1)
            .evaluate_batch_in(&context)
            .unwrap();

        assert_eq!(batch.outputs.len(), expected.len());
        assert_eq!(batch.stats.kernel_launch_count, 3);
        assert_eq!(batch.stats.readback_count, 1);
        assert_eq!(batch.stats.transfer_bytes, expected.len() * 168);
        for (index, (output, expected)) in batch.outputs.iter().zip(expected).enumerate() {
            let actual = output.tensor.owned_values[0];
            let difference = (actual - expected).abs();
            assert!(
                difference <= 1e-12 + 1e-12 * expected.abs(),
                "item {index}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.3e}"
            );
            assert_eq!(output.tensor.extents, vec![1, 1, 1, 1]);
            assert_eq!(output.stats.transfer_bytes, 168);
        }
        assert_eq!(context.stats().backend_cache_entries, 1);
    }

    #[test]
    fn ss_batch_pilot_keeps_multi_contraction_shells_on_the_compatibility_route() {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());
        let shell_i = Arc::new(
            Shell::try_new(
                0,
                0,
                2,
                2,
                0,
                Representation::Cart,
                arc_f64(&[1.0, 0.4]),
                arc_f64(&[0.6, 0.2, 0.1, 0.7]),
            )
            .unwrap(),
        );
        let shell_j = Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                arc_f64(&[0.8]),
                arc_f64(&[1.0]),
            )
            .unwrap(),
        );
        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_i.clone(), shell_j.clone()].into_boxed_slice()),
        )
        .unwrap();
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            ShellTuple::try_from_iter([shell_i, shell_j]).unwrap(),
            ExecutionOptions::default(),
        );

        assert!(ss_batch_inputs(&[request]).is_none());
    }

    #[test]
    fn batch_into_leaves_destination_unchanged_when_preflight_fails() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let valid = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let invalid_tuple = ShellTuple::try_from_iter([basis.shells()[0].clone()]).unwrap();
        let invalid = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            invalid_tuple,
            ExecutionOptions::default(),
        );
        let mut destination = vec![TypedEvaluationOutput::default(); 2];
        destination[0].bytes_written = 41;
        destination[1].bytes_written = 43;

        let error = BatchRequest::new([valid, invalid])
            .evaluate_batch_into(&mut destination)
            .unwrap_err();

        assert_eq!(error.kind(), FacadeErrorKind::Validation);
        assert_eq!(destination[0].bytes_written, 41);
        assert_eq!(destination[1].bytes_written, 43);
    }

    #[test]
    fn evaluation_context_is_shared_by_queries_and_preserves_results() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let context = EvaluationContext::new();
        let first = request.query_workspace_in(&context).unwrap();
        let second = request.query_workspace_in(&context).unwrap();

        assert!(Arc::ptr_eq(&first.executor, &second.executor));
        assert!(Arc::ptr_eq(
            &first.workspace_allocator,
            &second.workspace_allocator
        ));
        assert_eq!(
            first.evaluate().unwrap().tensor.owned_values,
            second.evaluate().unwrap().tensor.owned_values
        );
        let stats = context.stats();
        assert_eq!(stats.backend_cache_entries, 1);
        assert_eq!(stats.resident_metadata_entries, 1);
        assert!(stats.host_workspace_allocations >= 1);
        assert!(stats.host_workspace_reuses >= 1);
    }

    #[test]
    fn reusable_session_keeps_backend_and_workspace_state_private_but_reused() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let session = Session::new();

        let first = session.evaluate(&request).unwrap();
        let second = session.evaluate(&request).unwrap();

        assert_eq!(first.tensor.owned_values, second.tensor.owned_values);
        let stats = session.stats();
        assert_eq!(stats.backend_cache_entries, 1);
        assert!(stats.host_workspace_allocations >= 1);
        assert!(stats.host_workspace_reuses >= 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Shell-quartet batch surface (Task 34-F)
// ─────────────────────────────────────────────────────────────────────────────

/// A whole shell-quartet work list submitted as one request.
///
/// [`SessionRequest`] evaluates exactly one shell tuple, which is the right
/// shape for the compatibility API and the wrong shape for a Fock build: the
/// per-tuple route pays a kernel launch and a readback per quartet. This request
/// carries the entire list, so the backend can group it into launch classes and
/// dispatch once per class.
///
/// Scope: `int2e` in the spherical representation. Any other operator or
/// representation returns [`FacadeError::UnsupportedApi`] before any device work
/// — the batched kernel path is the plain Coulomb one, and quietly routing a
/// different operator through it would be worse than refusing.
///
/// No CubeCL type appears in this surface. `quartets` are indices into
/// `basis.shells()`, and results come back as ordinary `f64` AO blocks.
#[derive(Clone, Debug)]
pub struct QuartetBatchRequest<'basis> {
    operator: OperatorId,
    representation: Representation,
    basis: &'basis BasisSet,
    quartets: Vec<[u32; 4]>,
    tolerance: f64,
    options: ExecutionOptions,
}

/// Spherical AO blocks for a quartet batch, plus its execution statistics.
///
/// A claimed speed-up is only auditable if the launch and transfer counts that
/// produced it travel with the values, so [`Self::stats`] is not optional.
#[derive(Clone, Debug, PartialEq)]
pub struct QuartetBatchOutput {
    /// Concatenated spherical AO blocks, in the request's quartet order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where quartet `n`'s block starts in [`Self::values`].
    pub offsets: Vec<usize>,
    /// Submission-level counters for the whole batch.
    pub stats: BatchExecutionStats,
}

impl<'basis> QuartetBatchRequest<'basis> {
    /// Build a request over `quartets`, each a `[i, j, k, l]` of indices into
    /// `basis.shells()`.
    pub fn new(
        operator: OperatorId,
        representation: Representation,
        basis: &'basis BasisSet,
        quartets: impl IntoIterator<Item = [u32; 4]>,
        options: ExecutionOptions,
    ) -> Self {
        Self {
            operator,
            representation,
            basis,
            quartets: quartets.into_iter().collect(),
            tolerance: 0.0,
            options,
        }
    }

    /// Primitive-quartet screening tolerance.
    ///
    /// The default, `0.0`, is exact: it reproduces the unscreened arithmetic bit
    /// for bit. A positive value skips primitive quartets whose contribution
    /// scale factor does not exceed it, trading accuracy for work.
    #[must_use]
    pub fn tolerance(mut self, value: f64) -> Self {
        self.tolerance = value.max(0.0);
        self
    }

    pub fn operator(&self) -> OperatorId {
        self.operator
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn basis(&self) -> &'basis BasisSet {
        self.basis
    }

    pub fn quartets(&self) -> &[[u32; 4]] {
        &self.quartets
    }

    pub fn options(&self) -> &ExecutionOptions {
        &self.options
    }

    pub fn len(&self) -> usize {
        self.quartets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.quartets.is_empty()
    }

    /// Evaluate the whole list, allocating a fresh execution context.
    pub fn evaluate(self) -> Result<QuartetBatchOutput, FacadeError> {
        self.evaluate_in(&EvaluationContext::new())
    }

    /// Evaluate with a reusable context, so repeated Fock builds share the
    /// backend client rather than bootstrapping one per call.
    pub fn evaluate_in(
        self,
        context: &EvaluationContext,
    ) -> Result<QuartetBatchOutput, FacadeError> {
        let descriptor =
            Resolver::descriptor(self.operator).map_err(|error| FacadeError::UnsupportedApi {
                requested: error.to_string(),
            })?;
        // `int2e_sph` specifically, not merely the 2e family: the batched kernel
        // is the plain Coulomb one, so accepting `int2e_ip1_sph` here would hand
        // back undifferentiated integrals under a derivative operator's name.
        if descriptor.operator_symbol() != "int2e_sph" {
            return Err(FacadeError::UnsupportedApi {
                requested: format!(
                    "quartet-batch:operator:{} (only int2e_sph is batched)",
                    descriptor.operator_symbol()
                ),
            });
        }
        if self.representation != Representation::Spheric {
            return Err(FacadeError::UnsupportedApi {
                requested: format!(
                    "quartet-batch:representation:{} (only Spheric is batched)",
                    self.representation
                ),
            });
        }
        if let Some(aosym) = self.options.aosym {
            if aosym != cintx_core::AoSymmetry::S1 {
                return Err(FacadeError::UnsupportedAoSymmetry {
                    requested: aosym.to_string(),
                });
            }
        }
        if self.options.precision != PrecisionKind::F64 {
            return Err(FacadeError::UnsupportedApi {
                requested: "quartet-batch:precision (only F64 is batched)".to_owned(),
            });
        }

        let shells = batch_shells_from_basis(self.basis)?;
        for quartet in &self.quartets {
            for &index in quartet {
                if index as usize >= shells.len() {
                    return Err(FacadeError::Validation {
                        detail: format!(
                            "quartet-batch:shell-index {index} out of range (nbas={})",
                            shells.len()
                        ),
                    });
                }
            }
        }

        let submit_start = Instant::now();
        let batch = context
            .executor
            .evaluate_2e_quartets(
                &self.options.backend_intent,
                &shells,
                &self.quartets,
                cintx_cubecl::TwoEBatchOptions {
                    primitive_tolerance: self.tolerance,
                },
            )
            .map_err(FacadeError::from)?;
        let submit_ns = submit_start.elapsed().as_nanos() as u64;

        let stats = BatchExecutionStats {
            items_planned: self.quartets.len(),
            items_executed: batch.stats.quartets,
            // A bucket is one angular-momentum class; a chunk is one dispatch.
            // They coincided until Task 35-M1 merged every class sharing the
            // kernel's comptime signature into a single launch, so they are now
            // reported from the two counters the backend actually keeps apart.
            bucket_count: batch.stats.launch_classes,
            chunk_count: batch.stats.kernel_launch_count,
            kernel_launch_count: batch.stats.kernel_launch_count,
            readback_count: batch.stats.readback_count,
            transfer_bytes: batch.stats.transfer_bytes,
            pack_ns: 0,
            submit_ns,
            readback_ns: batch.stats.dispatch_ns,
            pilot_output_staging_allocations: 0,
            pilot_output_staging_reuses: 0,
            pilot_output_staging_growths: 0,
        };

        Ok(QuartetBatchOutput {
            values: batch.values,
            offsets: batch.offsets,
            stats,
        })
    }
}

/// Evaluate a whole shell-quartet work list. See [`QuartetBatchRequest`].
pub fn evaluate_shell_quartets(
    request: QuartetBatchRequest<'_>,
) -> Result<QuartetBatchOutput, FacadeError> {
    request.evaluate()
}

/// Evaluate a shell-quartet work list on a reusable context.
pub fn evaluate_shell_quartets_in(
    request: QuartetBatchRequest<'_>,
    context: &EvaluationContext,
) -> Result<QuartetBatchOutput, FacadeError> {
    request.evaluate_in(context)
}

/// Flatten a [`BasisSet`]'s AO shells into the backend's batch shell table.
///
/// `Shell::coefficients` is already primitive-major (`coeff[p * nctr + c]`),
/// which is the layout the batched kernel reads, so no transpose happens here —
/// the raw `atm`/`bas`/`env` route is the one that has to transpose (WR-03).
fn batch_shells_from_basis(basis: &BasisSet) -> Result<Vec<cintx_cubecl::BatchShell>, FacadeError> {
    let atoms = basis.atoms();
    let mut shells = Vec::with_capacity(basis.shells().len());
    for shell in basis.shells() {
        let atom = atoms
            .get(shell.atom_index as usize)
            .ok_or_else(|| FacadeError::Validation {
                detail: format!(
                    "quartet-batch:shell references atom {} of {}",
                    shell.atom_index,
                    atoms.len()
                ),
            })?;
        shells.push(cintx_cubecl::BatchShell {
            l: shell.ang_momentum,
            nprim: u32::from(shell.nprim),
            nctr: u32::from(shell.nctr),
            exponents: shell.exponents.to_vec(),
            coefficients: shell.coefficients.to_vec(),
            center: atom.coord_bohr,
        });
    }
    Ok(shells)
}
