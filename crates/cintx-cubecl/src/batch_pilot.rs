//! Narrow lane-per-tuple CubeCL pilot used to validate the batched launch shape.
//!
//! This deliberately covers only single-contraction Cartesian s-s overlap and
//! kinetic integrals. It is a correctness reference for grid-stride batching, not a fallback
//! for the general 1e launcher. General shells need transposed per-item scratch and
//! packed descriptors before they can use the same submission path.

use crate::backend::ResolvedBackend;
use crate::math::boys::boys_f0_f64;
use cintx_core::cintxRsError;
use cintx_runtime::BackendIntent;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const SQRTPI: f64 = 1.772_453_850_905_515_9;

/// One independent single-contraction Cartesian s-s tuple.
///
/// Each coefficient slice is paired with the exponent slice at the same index.
/// The narrow pilot intentionally accepts arbitrary primitive counts but exactly
/// one contraction per shell; higher contractions continue through the scalar
/// compatibility route until their output layout has a dedicated batch kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapSsInput {
    pub exponents_i: Arc<[f64]>,
    pub exponents_j: Arc<[f64]>,
    pub coefficients_i: Arc<[f64]>,
    pub coefficients_j: Arc<[f64]>,
    pub center_i: [f64; 3],
    pub center_j: [f64; 3],
}

/// One primitive, single-contraction Cartesian `(s s | s s)` tuple.
///
/// This deliberately admits exactly one primitive and one contraction on each
/// shell.  The safe facade leaves every contracted or angular 2e request on
/// the established scalar compatibility route until its descriptor/output
/// layout has an independently verified batch implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct EriSsssInput {
    pub exponents: [f64; 4],
    pub coefficients: [f64; 4],
    pub centers: [[f64; 3]; 4],
}

impl OverlapSsInput {
    pub fn primitive_counts(&self) -> (usize, usize) {
        (
            self.exponents_i.len().min(self.coefficients_i.len()),
            self.exponents_j.len().min(self.coefficients_j.len()),
        )
    }
}

/// Results and host wall-clock timings for one collective pilot submission.
///
/// `submit_ns` covers host descriptor packing, buffer creation, and launch
/// encoding. `readback_ns` covers the one collective `ComputeClient::read`
/// call, which is also the synchronization point on asynchronous backends.
/// These are deliberately *not* reported as device timestamps: they provide a
/// portable Phase-0 control-plane measurement until backend timestamp queries
/// are wired through CubeCL.
#[derive(Clone, Debug, PartialEq)]
pub struct SsBatchChunkOutput {
    pub chunks: Vec<Vec<f64>>,
    /// Exact bytes submitted for every descriptor/output table in each chunk.
    pub chunk_transfer_bytes: Vec<usize>,
    pub submit_ns: u64,
    pub readback_ns: u64,
    /// Output-only device staging allocations made for this submission.
    /// Dynamic descriptor tables are intentionally excluded because they are
    /// uploaded fresh for every request.
    pub output_staging_allocations: usize,
    /// Output chunks served by a same-or-larger retained staging slot.
    pub output_staging_reuses: usize,
    /// Fresh output allocations required because an existing slot was too
    /// small for the current chunk.
    pub output_staging_growths: usize,
}

/// Reusable output-only storage for the verified Cartesian s-s batch pilot.
///
/// The arena is keyed by the exact query-time [`BackendIntent`], which is also
/// the key of the executor's retained client. It deliberately does **not**
/// retain descriptor tables: their values vary per request and CubeCL 0.10's
/// public `ComputeClient` API has no raw-handle overwrite operation. They are
/// therefore uploaded afresh on every submission, while an output slot is
/// reused whenever its prior capacity covers the current chunk.
///
/// Callers must serialize a submission while borrowing this arena. Reusing an
/// output handle before its collective readback completes would otherwise make
/// concurrent requests alias their caller-visible staging results.
#[derive(Debug, Default)]
pub(crate) struct PilotOutputArena {
    slots: HashMap<BackendIntent, Vec<PilotOutputSlot>>,
    allocations: usize,
    reuses: usize,
    growths: usize,
    retained_bytes: usize,
    peak_retained_bytes: usize,
}

#[derive(Debug)]
struct PilotOutputSlot {
    handle: Handle,
    capacity_elements: usize,
}

/// Observable accounting for retained pilot output storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PilotOutputArenaStats {
    pub allocations: usize,
    pub reuses: usize,
    pub growths: usize,
    pub retained_bytes: usize,
    pub peak_retained_bytes: usize,
}

impl PilotOutputArena {
    pub(crate) fn stats(&self) -> PilotOutputArenaStats {
        PilotOutputArenaStats {
            allocations: self.allocations,
            reuses: self.reuses,
            growths: self.growths,
            retained_bytes: self.retained_bytes,
            peak_retained_bytes: self.peak_retained_bytes,
        }
    }

    fn output_handle<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        intent: &BackendIntent,
        slot_index: usize,
        required_elements: usize,
    ) -> Handle {
        debug_assert!(required_elements > 0);
        if let Some(handle) = self
            .slots
            .get(intent)
            .and_then(|slots| slots.get(slot_index))
            .filter(|slot| slot.capacity_elements >= required_elements)
            .map(|slot| slot.handle.clone())
        {
            self.reuses = self.reuses.saturating_add(1);
            return handle;
        }

        let new_bytes = required_elements.saturating_mul(std::mem::size_of::<f64>());
        let old_capacity = self
            .slots
            .get(intent)
            .and_then(|slots| slots.get(slot_index))
            .map(|slot| slot.capacity_elements);
        let handle = client.empty(new_bytes);
        let slots = self.slots.entry(intent.clone()).or_default();
        match slots.get_mut(slot_index) {
            Some(slot) => {
                self.growths = self.growths.saturating_add(1);
                self.retained_bytes = self
                    .retained_bytes
                    .saturating_sub(
                        old_capacity
                            .expect("existing output slot has capacity")
                            .saturating_mul(std::mem::size_of::<f64>()),
                    )
                    .saturating_add(new_bytes);
                *slot = PilotOutputSlot {
                    handle: handle.clone(),
                    capacity_elements: required_elements,
                };
            }
            None => {
                debug_assert_eq!(slot_index, slots.len());
                slots.push(PilotOutputSlot {
                    handle: handle.clone(),
                    capacity_elements: required_elements,
                });
                self.retained_bytes = self.retained_bytes.saturating_add(new_bytes);
            }
        }
        self.allocations = self.allocations.saturating_add(1);
        self.peak_retained_bytes = self.peak_retained_bytes.max(self.retained_bytes);
        handle
    }
}

/// Grid-stride, lane-per-tuple s-s kernel.
///
/// Each tuple owns exactly one final output slot. The host launcher proves that
/// every input table has the required `item_count`-scaled length; the loop guard
/// still makes over/under-provisioned launch geometry safe.
#[cube(launch, launch_unchecked)]
fn ss_grid_stride_kernel<F: Float + CubeElement>(
    exponents_i: &Array<F>,
    coefficients_i: &Array<F>,
    exponents_j: &Array<F>,
    coefficients_j: &Array<F>,
    centers: &Array<F>,
    primitive_counts: &Array<u32>,
    output: &mut Array<F>,
    item_count: usize,
    primitive_stride_i: usize,
    primitive_stride_j: usize,
    sqrtpi: F,
    pi_const: F,
    kinetic: u32,
) {
    let stride = (CUBE_COUNT_X * CUBE_DIM_X) as usize;
    let mut item = ABSOLUTE_POS;
    while item < item_count {
        let center_offset = item * 6;
        let dx = centers[center_offset] - centers[center_offset + 3];
        let dy = centers[center_offset + 1] - centers[center_offset + 4];
        let dz = centers[center_offset + 2] - centers[center_offset + 5];
        let distance_squared = dx * dx + dy * dy + dz * dz;
        let offset_i = item * primitive_stride_i;
        let offset_j = item * primitive_stride_j;
        let count_offset = item * 2;
        let count_i = primitive_counts[count_offset] as usize;
        let count_j = primitive_counts[count_offset + 1] as usize;
        let mut value = F::new(0.0);
        let mut primitive_i = 0usize;
        while primitive_i < count_i {
            let ai = exponents_i[offset_i + primitive_i];
            let ci = coefficients_i[offset_i + primitive_i];
            let mut primitive_j = 0usize;
            while primitive_j < count_j {
                let aj = exponents_j[offset_j + primitive_j];
                let cj = coefficients_j[offset_j + primitive_j];
                let zeta = ai + aj;
                let reduced_exponent = ai * aj / zeta;
                let overlap =
                    ci * cj * F::exp(-reduced_exponent * distance_squared) * sqrtpi * pi_const
                        / (zeta * F::sqrt(zeta) * F::new(4.0) * pi_const);
                if kinetic == 0 {
                    value += overlap;
                } else {
                    value += overlap
                        * reduced_exponent
                        * (F::new(3.0) - F::new(2.0) * reduced_exponent * distance_squared);
                }
                primitive_j += 1;
            }
            primitive_i += 1;
        }
        output[item] = value;
        item += stride;
    }
}

/// Grid-stride f64 kernel for one primitive Cartesian `(s s | s s)` ERI per
/// tuple.  `boys_f0_f64` is the Cody f64 device routine: do not replace it
/// with generic Boys machinery or `Float::erf`, whose coefficients/lowering
/// are insufficient for this compatibility-gated pilot.
#[cube(launch, launch_unchecked)]
fn eri_ssss_grid_stride_kernel(
    exponents: &Array<f64>,
    coefficients: &Array<f64>,
    centers: &Array<f64>,
    output: &mut Array<f64>,
    item_count: usize,
) {
    let stride = (CUBE_COUNT_X * CUBE_DIM_X) as usize;
    let mut item = ABSOLUTE_POS;
    while item < item_count {
        let exponent_offset = item * 4;
        let center_offset = item * 12;
        let a = exponents[exponent_offset];
        let b = exponents[exponent_offset + 1];
        let c = exponents[exponent_offset + 2];
        let d = exponents[exponent_offset + 3];
        let p = a + b;
        let q = c + d;
        let mu = a * b / p;
        let nu = c * d / q;

        let ax = centers[center_offset];
        let ay = centers[center_offset + 1];
        let az = centers[center_offset + 2];
        let bx = centers[center_offset + 3];
        let by = centers[center_offset + 4];
        let bz = centers[center_offset + 5];
        let cx = centers[center_offset + 6];
        let cy = centers[center_offset + 7];
        let cz = centers[center_offset + 8];
        let dx = centers[center_offset + 9];
        let dy = centers[center_offset + 10];
        let dz = centers[center_offset + 11];

        let ab2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);
        let cd2 = (cx - dx) * (cx - dx) + (cy - dy) * (cy - dy) + (cz - dz) * (cz - dz);
        let px = (a * ax + b * bx) / p;
        let py = (a * ay + b * by) / p;
        let pz = (a * az + b * bz) / p;
        let qx = (c * cx + d * dx) / q;
        let qy = (c * cy + d * dy) / q;
        let qz = (c * cz + d * dz) / q;
        let rho = p * q / (p + q);
        let pq2 = (px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz);
        // The scalar `int2e_cart` path uses the generated 2e common factor
        // with one `common_fac_sp(0) = 1/sqrt(4π)` per s shell.  Fold that
        // four-shell convention into this closed-form primitive prefactor so
        // the pilot matches the safe compatibility route exactly.
        let prefactor = f64::sqrt(std::f64::consts::PI) / (8.0 * p * q * f64::sqrt(p + q));
        output[item] = coefficients[exponent_offset]
            * coefficients[exponent_offset + 1]
            * coefficients[exponent_offset + 2]
            * coefficients[exponent_offset + 3]
            * f64::exp(-(mu * ab2 + nu * cd2))
            * prefactor
            * boys_f0_f64(rho * pq2);
        item += stride;
    }
}

/// Execute the grid-stride pilot for a runtime-specific CubeCL client.
#[cfg(test)]
pub fn run_overlap_ss_batch_device<R: Runtime>(
    client: &ComputeClient<R>,
    inputs: &[OverlapSsInput],
) -> Vec<f64> {
    if inputs.is_empty() {
        return Vec::new();
    }
    run_overlap_ss_batch_chunks_device(client, &[inputs])
        .chunks
        .into_iter()
        .next()
        .expect("one input batch produces one output batch")
}

/// Submit every disjoint batch chunk before one collective device-to-host readback.
///
/// Each chunk owns distinct output storage. `ComputeClient::read` accepts every
/// output handle at once, so this has one synchronization/readback boundary even
/// when memory planning requires several launches.
pub fn run_overlap_ss_batch_chunks_device<R: Runtime>(
    client: &ComputeClient<R>,
    chunks: &[&[OverlapSsInput]],
) -> SsBatchChunkOutput {
    let mut arena = PilotOutputArena::default();
    run_ss_batch_chunks_device(client, chunks, false, &BackendIntent::default(), &mut arena)
}

/// Submit every planned kinetic s-s chunk before one collective readback.
pub fn run_kinetic_ss_batch_chunks_device<R: Runtime>(
    client: &ComputeClient<R>,
    chunks: &[&[OverlapSsInput]],
) -> SsBatchChunkOutput {
    let mut arena = PilotOutputArena::default();
    run_ss_batch_chunks_device(client, chunks, true, &BackendIntent::default(), &mut arena)
}

/// Submit every primitive `(s s | s s)` chunk before one collective readback.
pub fn run_eri_ssss_batch_chunks_device<R: Runtime>(
    client: &ComputeClient<R>,
    chunks: &[&[EriSsssInput]],
) -> SsBatchChunkOutput {
    let submit_started = Instant::now();
    let mut output_handles = Vec::with_capacity(chunks.len());
    let mut output_lengths = Vec::with_capacity(chunks.len());
    let mut output_indices = Vec::with_capacity(chunks.len());
    let mut chunk_transfer_bytes = vec![0; chunks.len()];
    let mut outputs: Vec<Option<Vec<f64>>> = (0..chunks.len()).map(|_| None).collect();

    for (chunk_index, inputs) in chunks.iter().enumerate() {
        if inputs.is_empty() {
            outputs[chunk_index] = Some(Vec::new());
            continue;
        }
        let mut exponents = Vec::with_capacity(inputs.len() * 4);
        let mut coefficients = Vec::with_capacity(inputs.len() * 4);
        let mut centers = Vec::with_capacity(inputs.len() * 12);
        for input in *inputs {
            exponents.extend(input.exponents);
            coefficients.extend(input.coefficients);
            for center in input.centers {
                centers.extend(center);
            }
        }
        let exponent_handle = client.create_from_slice(f64::as_bytes(&exponents));
        let coefficient_handle = client.create_from_slice(f64::as_bytes(&coefficients));
        let center_handle = client.create_from_slice(f64::as_bytes(&centers));
        let output_handle = client.empty(inputs.len() * std::mem::size_of::<f64>());
        chunk_transfer_bytes[chunk_index] = exponents
            .len()
            .saturating_add(coefficients.len())
            .saturating_add(centers.len())
            .saturating_add(inputs.len())
            .saturating_mul(std::mem::size_of::<f64>());
        let cube_dim = 64u32;
        let cube_count = (inputs.len() as u32).div_ceil(cube_dim).clamp(1, 64);

        // SAFETY: every table is packed at its fixed tuple stride and the
        // grid-stride guard bounds all accesses by `item_count`.
        unsafe {
            eri_ssss_grid_stride_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(exponent_handle, exponents.len()),
                ArrayArg::from_raw_parts(coefficient_handle, coefficients.len()),
                ArrayArg::from_raw_parts(center_handle, centers.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), inputs.len()),
                inputs.len(),
            );
        }
        output_handles.push(output_handle);
        output_lengths.push(inputs.len());
        output_indices.push(chunk_index);
    }

    let submit_ns = elapsed_ns(submit_started);
    let readback_started = Instant::now();
    for ((bytes, len), chunk_index) in client
        .read(output_handles)
        .into_iter()
        .zip(output_lengths)
        .zip(output_indices)
    {
        outputs[chunk_index] = Some(f64::from_bytes(&bytes)[..len].to_vec());
    }
    SsBatchChunkOutput {
        chunks: outputs
            .into_iter()
            .map(|output| output.expect("every input chunk receives an output vector"))
            .collect(),
        chunk_transfer_bytes,
        submit_ns,
        readback_ns: elapsed_ns(readback_started),
        output_staging_allocations: 0,
        output_staging_reuses: 0,
        output_staging_growths: 0,
    }
}

fn run_ss_batch_chunks_device<R: Runtime>(
    client: &ComputeClient<R>,
    chunks: &[&[OverlapSsInput]],
    kinetic: bool,
    intent: &BackendIntent,
    output_arena: &mut PilotOutputArena,
) -> SsBatchChunkOutput {
    let arena_before = output_arena.stats();
    let submit_started = Instant::now();
    let mut output_handles = Vec::with_capacity(chunks.len());
    let mut output_lengths = Vec::with_capacity(chunks.len());
    let mut output_indices = Vec::with_capacity(chunks.len());
    let mut chunk_transfer_bytes = vec![0; chunks.len()];
    let mut outputs: Vec<Option<Vec<f64>>> = (0..chunks.len()).map(|_| None).collect();

    for (chunk_index, inputs) in chunks.iter().enumerate() {
        if inputs.is_empty() {
            outputs[chunk_index] = Some(Vec::new());
            continue;
        }
        let max_primitives_i = inputs
            .iter()
            .map(|input| input.primitive_counts().0)
            .max()
            .unwrap_or(0);
        let max_primitives_j = inputs
            .iter()
            .map(|input| input.primitive_counts().1)
            .max()
            .unwrap_or(0);
        let mut exponents_i = vec![0.0; inputs.len() * max_primitives_i];
        let mut coefficients_i = vec![0.0; inputs.len() * max_primitives_i];
        let mut exponents_j = vec![0.0; inputs.len() * max_primitives_j];
        let mut coefficients_j = vec![0.0; inputs.len() * max_primitives_j];
        let mut centers = Vec::with_capacity(inputs.len() * 6);
        let mut primitive_counts = Vec::with_capacity(inputs.len() * 2);
        for (item, input) in inputs.iter().enumerate() {
            let (count_i, count_j) = input.primitive_counts();
            let offset_i = item * max_primitives_i;
            let offset_j = item * max_primitives_j;
            exponents_i[offset_i..offset_i + count_i].copy_from_slice(&input.exponents_i);
            coefficients_i[offset_i..offset_i + count_i].copy_from_slice(&input.coefficients_i);
            exponents_j[offset_j..offset_j + count_j].copy_from_slice(&input.exponents_j);
            coefficients_j[offset_j..offset_j + count_j].copy_from_slice(&input.coefficients_j);
            centers.extend(input.center_i);
            centers.extend(input.center_j);
            primitive_counts.extend([
                u32::try_from(count_i).expect("validated shell primitive count fits u32"),
                u32::try_from(count_j).expect("validated shell primitive count fits u32"),
            ]);
        }

        let exponent_i_handle = client.create_from_slice(f64::as_bytes(&exponents_i));
        let coefficient_i_handle = client.create_from_slice(f64::as_bytes(&coefficients_i));
        let exponent_j_handle = client.create_from_slice(f64::as_bytes(&exponents_j));
        let coefficient_j_handle = client.create_from_slice(f64::as_bytes(&coefficients_j));
        let center_handle = client.create_from_slice(f64::as_bytes(&centers));
        let primitive_counts_handle = client.create_from_slice(u32::as_bytes(&primitive_counts));
        let output_handle = output_arena.output_handle(client, intent, chunk_index, inputs.len());
        chunk_transfer_bytes[chunk_index] = exponents_i
            .len()
            .saturating_add(coefficients_i.len())
            .saturating_add(exponents_j.len())
            .saturating_add(coefficients_j.len())
            .saturating_add(centers.len())
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_add(
                primitive_counts
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(inputs.len().saturating_mul(std::mem::size_of::<f64>()));
        let cube_dim = 64u32;
        let cube_count = (inputs.len() as u32).div_ceil(cube_dim).clamp(1, 64);

        // SAFETY: Every table has exactly the item-count-scaled length indexed by
        // the kernel. `item < item_count` bounds all accesses and output slots.
        unsafe {
            ss_grid_stride_kernel::launch_unchecked::<f64, R>(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(exponent_i_handle, exponents_i.len()),
                ArrayArg::from_raw_parts(coefficient_i_handle, coefficients_i.len()),
                ArrayArg::from_raw_parts(exponent_j_handle, exponents_j.len()),
                ArrayArg::from_raw_parts(coefficient_j_handle, coefficients_j.len()),
                ArrayArg::from_raw_parts(center_handle, centers.len()),
                ArrayArg::from_raw_parts(primitive_counts_handle, primitive_counts.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), inputs.len()),
                inputs.len(),
                max_primitives_i,
                max_primitives_j,
                SQRTPI,
                std::f64::consts::PI,
                u32::from(kinetic),
            );
        }
        output_handles.push(output_handle);
        output_lengths.push(inputs.len());
        output_indices.push(chunk_index);
    }

    let submit_ns = elapsed_ns(submit_started);
    let readback_started = Instant::now();
    for ((bytes, len), chunk_index) in client
        .read(output_handles)
        .into_iter()
        .zip(output_lengths)
        .zip(output_indices)
    {
        outputs[chunk_index] = Some(f64::from_bytes(&bytes)[..len].to_vec());
    }
    let arena_after = output_arena.stats();
    SsBatchChunkOutput {
        chunks: outputs
            .into_iter()
            .map(|output| output.expect("every input chunk receives an output vector"))
            .collect(),
        chunk_transfer_bytes,
        submit_ns,
        readback_ns: elapsed_ns(readback_started),
        output_staging_allocations: arena_after
            .allocations
            .saturating_sub(arena_before.allocations),
        output_staging_reuses: arena_after.reuses.saturating_sub(arena_before.reuses),
        output_staging_growths: arena_after.growths.saturating_sub(arena_before.growths),
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Submit one lane-per-tuple grid-stride launch over the complete pilot batch.
pub fn run_overlap_ss_batch(
    backend: &ResolvedBackend,
    inputs: &[OverlapSsInput],
) -> Result<Vec<f64>, cintxRsError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    Ok(run_overlap_ss_batch_chunks(backend, &[inputs])
        .chunks
        .into_iter()
        .next()
        .expect("one input batch produces one output batch"))
}

/// Backend-generic collective submission for all planned overlap chunks.
pub fn run_overlap_ss_batch_chunks(
    backend: &ResolvedBackend,
    chunks: &[&[OverlapSsInput]],
) -> SsBatchChunkOutput {
    let mut arena = PilotOutputArena::default();
    run_ss_batch_chunks_with_output_arena(
        backend,
        chunks,
        false,
        &BackendIntent::default(),
        &mut arena,
    )
}

/// Backend-generic collective submission for all planned kinetic s-s chunks.
pub fn run_kinetic_ss_batch_chunks(
    backend: &ResolvedBackend,
    chunks: &[&[OverlapSsInput]],
) -> SsBatchChunkOutput {
    let mut arena = PilotOutputArena::default();
    run_ss_batch_chunks_with_output_arena(
        backend,
        chunks,
        true,
        &BackendIntent::default(),
        &mut arena,
    )
}

/// Backend-generic collective submission that retains only output staging in
/// the executor-owned arena. Descriptor inputs remain per-request uploads.
pub(crate) fn run_ss_batch_chunks_with_output_arena(
    backend: &ResolvedBackend,
    chunks: &[&[OverlapSsInput]],
    kinetic: bool,
    intent: &BackendIntent,
    output_arena: &mut PilotOutputArena,
) -> SsBatchChunkOutput {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_ss_batch_chunks_device(client, chunks, kinetic, intent, output_arena)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_ss_batch_chunks_device(client, chunks, kinetic, intent, output_arena)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_ss_batch_chunks_device(client, chunks, kinetic, intent, output_arena)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_ss_batch_chunks_device(client, chunks, kinetic, intent, output_arena)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_ss_batch_chunks_device(client, chunks, kinetic, intent, output_arena)
        }
    }
}

/// Backend-generic collective submission for the primitive Cartesian 2e pilot.
pub fn run_eri_ssss_batch_chunks(
    backend: &ResolvedBackend,
    chunks: &[&[EriSsssInput]],
) -> SsBatchChunkOutput {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_eri_ssss_batch_chunks_device(client, chunks),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_eri_ssss_batch_chunks_device(client, chunks),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_eri_ssss_batch_chunks_device(client, chunks),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_eri_ssss_batch_chunks_device(client, chunks),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_eri_ssss_batch_chunks_device(client, chunks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::boys::boys_gamma_inc_host;
    use cubecl::cpu::CpuRuntime;

    fn input(
        exponents_i: &[f64],
        coefficients_i: &[f64],
        exponents_j: &[f64],
        coefficients_j: &[f64],
        center_i: [f64; 3],
        center_j: [f64; 3],
    ) -> OverlapSsInput {
        OverlapSsInput {
            exponents_i: Arc::from(exponents_i.to_vec().into_boxed_slice()),
            exponents_j: Arc::from(exponents_j.to_vec().into_boxed_slice()),
            coefficients_i: Arc::from(coefficients_i.to_vec().into_boxed_slice()),
            coefficients_j: Arc::from(coefficients_j.to_vec().into_boxed_slice()),
            center_i,
            center_j,
        }
    }

    fn scalar_overlap(input: &OverlapSsInput) -> f64 {
        let dx = input.center_i[0] - input.center_j[0];
        let dy = input.center_i[1] - input.center_j[1];
        let dz = input.center_i[2] - input.center_j[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;
        input
            .exponents_i
            .iter()
            .zip(input.coefficients_i.iter())
            .flat_map(|(&ai, &ci)| {
                input
                    .exponents_j
                    .iter()
                    .zip(input.coefficients_j.iter())
                    .map(move |(&aj, &cj)| {
                        let zeta = ai + aj;
                        ci * cj
                            * (-ai * aj / zeta * distance_squared).exp()
                            * SQRTPI
                            * std::f64::consts::PI
                            / (zeta * zeta.sqrt() * 4.0 * std::f64::consts::PI)
                    })
            })
            .sum()
    }

    fn scalar_kinetic(input: &OverlapSsInput) -> f64 {
        let dx = input.center_i[0] - input.center_j[0];
        let dy = input.center_i[1] - input.center_j[1];
        let dz = input.center_i[2] - input.center_j[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;
        input
            .exponents_i
            .iter()
            .zip(input.coefficients_i.iter())
            .flat_map(|(&ai, &ci)| {
                input
                    .exponents_j
                    .iter()
                    .zip(input.coefficients_j.iter())
                    .map(move |(&aj, &cj)| {
                        let zeta = ai + aj;
                        let reduced_exponent = ai * aj / zeta;
                        let overlap = ci
                            * cj
                            * (-reduced_exponent * distance_squared).exp()
                            * SQRTPI
                            * std::f64::consts::PI
                            / (zeta * zeta.sqrt() * 4.0 * std::f64::consts::PI);
                        overlap
                            * reduced_exponent
                            * (3.0 - 2.0 * reduced_exponent * distance_squared)
                    })
            })
            .sum()
    }

    fn eri_input(
        exponents: [f64; 4],
        coefficients: [f64; 4],
        centers: [[f64; 3]; 4],
    ) -> EriSsssInput {
        EriSsssInput {
            exponents,
            coefficients,
            centers,
        }
    }

    fn scalar_eri_ssss(input: &EriSsssInput) -> f64 {
        let [a, b, c, d] = input.exponents;
        let [ca, cb, cc, cd] = input.coefficients;
        let [a_center, b_center, c_center, d_center] = input.centers;
        let p = a + b;
        let q = c + d;
        let mu = a * b / p;
        let nu = c * d / q;
        let ab2 = squared_distance(a_center, b_center);
        let cd2 = squared_distance(c_center, d_center);
        let p_center = weighted_center(a, a_center, b, b_center, p);
        let q_center = weighted_center(c, c_center, d, d_center, q);
        let rho = p * q / (p + q);
        let boys = boys_gamma_inc_host::<f64>(rho * squared_distance(p_center, q_center), 0)[0];
        ca * cb
            * cc
            * cd
            * (-(mu * ab2 + nu * cd2)).exp()
            * (std::f64::consts::PI.sqrt() / (8.0 * p * q * (p + q).sqrt()))
            * boys
    }

    fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
        (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
    }

    fn weighted_center(
        left_exponent: f64,
        left: [f64; 3],
        right_exponent: f64,
        right: [f64; 3],
        sum: f64,
    ) -> [f64; 3] {
        [
            (left_exponent * left[0] + right_exponent * right[0]) / sum,
            (left_exponent * left[1] + right_exponent * right[1]) / sum,
            (left_exponent * left[2] + right_exponent * right[2]) / sum,
        ]
    }

    #[test]
    fn grid_stride_overlap_matches_scalar_for_more_items_than_one_cube() {
        let inputs: Vec<_> = (0..129)
            .map(|index| {
                let value = index as f64;
                input(
                    &[0.4 + value * 0.001],
                    &[0.8],
                    &[0.7 + value * 0.002],
                    &[0.6],
                    [0.01 * value, -0.02 * value, 0.03 * value],
                    [0.3, -0.4, 1.2],
                )
            })
            .collect();
        let client = CpuRuntime::client(&Default::default());
        let actual = run_overlap_ss_batch_device(&client, &inputs);

        assert_eq!(actual.len(), inputs.len());
        for (index, (actual, input)) in actual.iter().zip(&inputs).enumerate() {
            let expected = scalar_overlap(input);
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "item {index}: actual={actual:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn collective_readback_preserves_disjoint_chunk_boundaries() {
        let inputs = [
            input(
                &[0.4],
                &[0.8],
                &[0.7],
                &[0.6],
                [0.0, 0.1, 0.2],
                [0.3, -0.4, 1.2],
            ),
            input(
                &[0.5],
                &[0.9],
                &[0.6],
                &[0.7],
                [0.2, -0.1, 0.4],
                [0.3, -0.4, 1.2],
            ),
            input(
                &[0.6],
                &[0.7],
                &[0.5],
                &[0.8],
                [-0.3, 0.2, -0.1],
                [0.3, -0.4, 1.2],
            ),
        ];
        let client = CpuRuntime::client(&Default::default());
        let actual = run_overlap_ss_batch_chunks_device(&client, &[&inputs[..1], &inputs[1..]]);

        assert_eq!(
            actual.chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 2]
        );
        for (index, (actual, input)) in actual.chunks.into_iter().flatten().zip(&inputs).enumerate()
        {
            let expected = scalar_overlap(input);
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "item {index}: actual={actual:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn output_arena_reuses_same_or_smaller_chunks_and_grows_without_losing_parity() {
        let client = CpuRuntime::client(&Default::default());
        let intent = BackendIntent::default();
        let mut arena = PilotOutputArena::default();
        let first = [
            input(
                &[0.4],
                &[0.8],
                &[0.7],
                &[0.6],
                [0.0, 0.1, 0.2],
                [0.3, -0.4, 1.2],
            ),
            input(
                &[0.5],
                &[0.9],
                &[0.6],
                &[0.7],
                [0.2, -0.1, 0.4],
                [0.3, -0.4, 1.2],
            ),
        ];
        let first_output =
            run_ss_batch_chunks_device(&client, &[&first], false, &intent, &mut arena);
        assert_eq!(first_output.output_staging_allocations, 1);
        assert_eq!(first_output.output_staging_reuses, 0);
        assert_eq!(first_output.output_staging_growths, 0);
        for (actual, input) in first_output.chunks.into_iter().flatten().zip(&first) {
            assert!((actual - scalar_overlap(input)).abs() <= 1e-12);
        }

        let second = [input(
            &[0.9],
            &[0.4],
            &[0.3],
            &[1.1],
            [-0.2, 0.3, -0.1],
            [0.5, 0.4, -0.7],
        )];
        let second_output =
            run_ss_batch_chunks_device(&client, &[&second], false, &intent, &mut arena);
        assert_eq!(second_output.output_staging_allocations, 0);
        assert_eq!(second_output.output_staging_reuses, 1);
        assert_eq!(second_output.output_staging_growths, 0);
        assert!((second_output.chunks[0][0] - scalar_overlap(&second[0])).abs() <= 1e-12);

        let third = [second[0].clone(), first[0].clone(), first[1].clone()];
        let third_output =
            run_ss_batch_chunks_device(&client, &[&third], false, &intent, &mut arena);
        assert_eq!(third_output.output_staging_allocations, 1);
        assert_eq!(third_output.output_staging_reuses, 0);
        assert_eq!(third_output.output_staging_growths, 1);
        for (actual, input) in third_output.chunks.into_iter().flatten().zip(&third) {
            assert!((actual - scalar_overlap(input)).abs() <= 1e-12);
        }

        assert_eq!(
            arena.stats(),
            PilotOutputArenaStats {
                allocations: 2,
                reuses: 1,
                growths: 1,
                retained_bytes: 3 * std::mem::size_of::<f64>(),
                peak_retained_bytes: 3 * std::mem::size_of::<f64>(),
            }
        );
    }

    #[test]
    fn grid_stride_kinetic_matches_scalar_for_disjoint_chunks() {
        let inputs: Vec<_> = (0..129)
            .map(|index| {
                let value = index as f64;
                input(
                    &[0.4 + value * 0.001],
                    &[0.8],
                    &[0.7 + value * 0.002],
                    &[0.6],
                    [0.01 * value, -0.02 * value, 0.03 * value],
                    [0.3, -0.4, 1.2],
                )
            })
            .collect();
        let client = CpuRuntime::client(&Default::default());
        let actual = run_kinetic_ss_batch_chunks_device(&client, &[&inputs[..64], &inputs[64..]]);

        assert_eq!(
            actual.chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![64, 65]
        );
        for (index, (actual, input)) in actual.chunks.into_iter().flatten().zip(&inputs).enumerate()
        {
            let expected = scalar_kinetic(input);
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "item {index}: actual={actual:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn grid_stride_eri_ssss_matches_host_boys_and_preserves_chunk_order() {
        let inputs: Vec<_> = (0..129)
            .map(|index| {
                let value = index as f64;
                eri_input(
                    [0.31 + 0.001 * value, 0.72, 0.43, 1.19],
                    [0.61, -0.37, 0.52, 0.91],
                    [
                        [0.01 * value, -0.02 * value, 0.03 * value],
                        [0.2, -0.5, 0.7],
                        [-0.4, 0.3, -0.1],
                        [0.6, 0.8, -0.2],
                    ],
                )
            })
            .collect();
        let client = CpuRuntime::client(&Default::default());
        let actual = run_eri_ssss_batch_chunks_device(&client, &[&inputs[..64], &inputs[64..]]);

        assert_eq!(
            actual.chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![64, 65]
        );
        assert_eq!(actual.chunk_transfer_bytes, vec![64 * 168, 65 * 168]);
        for (index, (actual, input)) in actual.chunks.into_iter().flatten().zip(&inputs).enumerate()
        {
            let expected = scalar_eri_ssss(input);
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "item {index}: actual={actual:.17e}, expected={expected:.17e}"
            );
        }
    }
}
