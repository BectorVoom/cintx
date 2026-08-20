//! Backend-neutral batch planning for CubeCL integral execution.
//!
//! A batch plan owns only control-plane metadata: item order, homogeneous
//! kernel buckets, final-output offsets, and disjoint chunk ranges.  Kernels
//! consume these tables later; no host integral computation belongs here.

use cintx_core::{PrecisionKind, Representation, cintxRsError};

/// Structural properties which may select a CubeCL specialization.
///
/// Dynamic numerical data is intentionally excluded so a basis change does
/// not create a new compilation variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelClass {
    pub family: &'static str,
    pub representation: Representation,
    pub precision: PrecisionKind,
    pub arity: u8,
    pub angular_momenta: Vec<u8>,
    pub nroots: u8,
    pub component_rank: u8,
}

/// One input item before offsets and chunks have been assigned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchItemRequest {
    pub kernel_class: KernelClass,
    pub output_elements: usize,
    pub scratch_bytes: usize,
}

/// One input item with a stable final-output range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchItem {
    pub input_index: usize,
    pub output_offset: usize,
    pub output_elements: usize,
    pub scratch_bytes: usize,
}

/// A homogeneous set of items eligible for one kernel variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchBucket {
    pub kernel_class: KernelClass,
    /// Original input indices; execution may reorder buckets but never output offsets.
    pub item_indices: Vec<usize>,
}

/// A disjoint executable item range inside one homogeneous bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchChunk {
    pub index: usize,
    pub bucket_index: usize,
    pub item_start: usize,
    pub item_count: usize,
    pub scratch_bytes: usize,
    pub output_bytes: usize,
}

/// Fully validated, backend-neutral plan for a batched submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchExecutionPlan {
    pub items: Vec<BatchItem>,
    pub buckets: Vec<BatchBucket>,
    pub chunks: Vec<BatchChunk>,
    pub total_output_elements: usize,
    pub total_output_bytes: usize,
}

impl BatchExecutionPlan {
    /// Construct a plan with disjoint, stable final-output offsets.
    ///
    /// `max_items_per_chunk` bounds descriptor work; `max_chunk_bytes` bounds
    /// the scratch plus final-layout bytes owned by a single kernel launch.
    pub fn build(
        requests: impl IntoIterator<Item = BatchItemRequest>,
        max_items_per_chunk: usize,
        max_chunk_bytes: usize,
    ) -> Result<Self, cintxRsError> {
        let requests: Vec<_> = requests.into_iter().collect();
        let max_items_per_chunk = max_items_per_chunk.max(1);
        let max_chunk_bytes = max_chunk_bytes.max(1);
        let mut items = Vec::with_capacity(requests.len());
        let mut buckets: Vec<BatchBucket> = Vec::new();
        let mut total_output_elements = 0usize;
        let mut total_output_bytes = 0usize;

        for (input_index, request) in requests.iter().enumerate() {
            let output_offset = total_output_elements;
            total_output_elements = total_output_elements
                .checked_add(request.output_elements)
                .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                    from: "batch_plan",
                    detail: "final output element count overflowed usize".to_owned(),
                })?;
            total_output_bytes = total_output_bytes
                .checked_add(output_bytes(
                    request.output_elements,
                    request.kernel_class.precision,
                )?)
                .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                    from: "batch_plan",
                    detail: "final output byte count overflowed usize".to_owned(),
                })?;
            items.push(BatchItem {
                input_index,
                output_offset,
                output_elements: request.output_elements,
                scratch_bytes: request.scratch_bytes,
            });

            if let Some(bucket) = buckets
                .iter_mut()
                .find(|bucket| bucket.kernel_class == request.kernel_class)
            {
                bucket.item_indices.push(input_index);
            } else {
                buckets.push(BatchBucket {
                    kernel_class: request.kernel_class.clone(),
                    item_indices: vec![input_index],
                });
            }
        }

        let mut chunks = Vec::new();
        for (bucket_index, bucket) in buckets.iter().enumerate() {
            let mut start = 0usize;
            while start < bucket.item_indices.len() {
                let mut count = 0usize;
                let mut scratch_bytes = 0usize;
                let mut output_bytes_total = 0usize;
                while start + count < bucket.item_indices.len() && count < max_items_per_chunk {
                    let item = &items[bucket.item_indices[start + count]];
                    let next_scratch =
                        scratch_bytes
                            .checked_add(item.scratch_bytes)
                            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                                from: "batch_plan",
                                detail: "batch scratch byte count overflowed usize".to_owned(),
                            })?;
                    let next_output = output_bytes_total
                        .checked_add(output_bytes(
                            item.output_elements,
                            bucket.kernel_class.precision,
                        )?)
                        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                            from: "batch_plan",
                            detail: "batch output byte count overflowed usize".to_owned(),
                        })?;
                    let total = next_scratch.checked_add(next_output).ok_or_else(|| {
                        cintxRsError::ChunkPlanFailed {
                            from: "batch_plan",
                            detail: "batch chunk byte count overflowed usize".to_owned(),
                        }
                    })?;
                    if total > max_chunk_bytes {
                        if count == 0 {
                            return Err(cintxRsError::MemoryLimitExceeded {
                                requested: total,
                                limit: max_chunk_bytes,
                            });
                        }
                        break;
                    }
                    scratch_bytes = next_scratch;
                    output_bytes_total = next_output;
                    count += 1;
                }
                chunks.push(BatchChunk {
                    index: chunks.len(),
                    bucket_index,
                    item_start: start,
                    item_count: count,
                    scratch_bytes,
                    output_bytes: output_bytes_total,
                });
                start += count;
            }
        }

        Ok(Self {
            items,
            buckets,
            chunks,
            total_output_elements,
            total_output_bytes,
        })
    }

    /// Returns the original input items covered by a disjoint chunk.
    pub fn chunk_items(&self, chunk: &BatchChunk) -> &[usize] {
        let bucket = &self.buckets[chunk.bucket_index];
        &bucket.item_indices[chunk.item_start..chunk.item_start + chunk.item_count]
    }
}

fn output_bytes(elements: usize, precision: PrecisionKind) -> Result<usize, cintxRsError> {
    elements
        .checked_mul(precision.element_size())
        .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(family: &'static str, precision: PrecisionKind) -> KernelClass {
        KernelClass {
            family,
            representation: Representation::Cart,
            precision,
            arity: 2,
            angular_momenta: vec![0, 0],
            nroots: 1,
            component_rank: 1,
        }
    }

    #[test]
    fn batch_plan_preserves_input_output_offsets_across_buckets() {
        let plan = BatchExecutionPlan::build(
            [
                BatchItemRequest {
                    kernel_class: class("1e", PrecisionKind::F64),
                    output_elements: 3,
                    scratch_bytes: 8,
                },
                BatchItemRequest {
                    kernel_class: class("2e", PrecisionKind::F64),
                    output_elements: 5,
                    scratch_bytes: 16,
                },
                BatchItemRequest {
                    kernel_class: class("1e", PrecisionKind::F64),
                    output_elements: 7,
                    scratch_bytes: 8,
                },
            ],
            8,
            1024,
        )
        .unwrap();

        assert_eq!(plan.total_output_elements, 15);
        assert_eq!(
            plan.items
                .iter()
                .map(|item| item.output_offset)
                .collect::<Vec<_>>(),
            vec![0, 3, 8]
        );
        assert_eq!(plan.buckets.len(), 2);
        assert_eq!(plan.buckets[0].item_indices, vec![0, 2]);
        assert_eq!(plan.buckets[1].item_indices, vec![1]);
    }

    #[test]
    fn batch_chunks_cover_every_item_once_within_limits() {
        let plan = BatchExecutionPlan::build(
            (0..5).map(|_| BatchItemRequest {
                kernel_class: class("1e", PrecisionKind::F64),
                output_elements: 2,
                scratch_bytes: 8,
            }),
            2,
            48,
        )
        .unwrap();

        let mut covered: Vec<_> = plan
            .chunks
            .iter()
            .flat_map(|chunk| plan.chunk_items(chunk))
            .copied()
            .collect();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 1, 2, 3, 4]);
        assert!(
            plan.chunks.iter().all(
                |chunk| chunk.item_count <= 2 && chunk.scratch_bytes + chunk.output_bytes <= 48
            )
        );
    }

    #[test]
    fn batch_plan_rejects_an_item_that_cannot_fit() {
        let error = BatchExecutionPlan::build(
            [BatchItemRequest {
                kernel_class: class("1e", PrecisionKind::F64),
                output_elements: 4,
                scratch_bytes: 8,
            }],
            1,
            32,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            cintxRsError::MemoryLimitExceeded {
                requested: 40,
                limit: 32
            }
        ));
    }
}
