use crate::options::{BackendCapabilityToken, BackendIntent, ExecutionOptions};
use cintx_core::cintxRsError;
use std::cmp;
use std::mem::{MaybeUninit, size_of};
use tracing::debug;

pub const DEFAULT_ALIGNMENT_BYTES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceQuery {
    pub bytes: usize,
    pub alignment: usize,
    pub required_bytes: usize,
    pub chunk_count: usize,
    pub work_units: usize,
    pub min_chunk_bytes: usize,
    pub fallback_reason: Option<&'static str>,
    pub chunks: Vec<ChunkInfo>,
    pub memory_limit_bytes: Option<usize>,
    pub chunk_size_override: Option<usize>,
    /// Backend selection intent captured at query time.
    pub backend_intent: BackendIntent,
    /// Backend capability token captured at query time.
    pub backend_capability_token: BackendCapabilityToken,
}

impl WorkspaceQuery {
    pub fn request(&self) -> WorkspaceRequest {
        WorkspaceRequest {
            required_bytes: self.required_bytes,
            alignment: self.alignment,
            work_units: self.work_units,
            min_chunk_bytes: self.min_chunk_bytes,
        }
    }

    /// Returns `true` only when all four contract fields match `opts`.
    ///
    /// Per D-08, backend policy drift between query and evaluate must be
    /// detected here so `evaluate()` can fail closed with a typed error.
    pub fn planning_matches(&self, opts: &ExecutionOptions) -> bool {
        self.memory_limit_bytes == opts.memory_limit_bytes
            && self.chunk_size_override == opts.chunk_size_override
            && self.backend_intent == opts.backend_intent
            && self.backend_capability_token == opts.backend_capability_token
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceRequest {
    pub required_bytes: usize,
    pub alignment: usize,
    pub work_units: usize,
    pub min_chunk_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInfo {
    pub index: usize,
    pub work_unit_start: usize,
    pub work_unit_count: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    pub chunks: Vec<ChunkInfo>,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkPlanner {
    limit_bytes: Option<usize>,
    chunk_size_override: Option<usize>,
}

impl ChunkPlanner {
    pub fn from_options(opts: &ExecutionOptions) -> Self {
        Self {
            limit_bytes: opts.memory_limit_bytes,
            chunk_size_override: opts.chunk_size_override,
        }
    }

    pub fn plan(&self, request: &WorkspaceRequest) -> Result<ChunkPlan, cintxRsError> {
        let effective_limit = self
            .limit_bytes
            .unwrap_or(request.required_bytes.max(request.alignment));
        let min_chunk_bytes = request.min_chunk_bytes.max(request.alignment);
        let work_units = request.work_units.max(1);

        debug!(
            required_bytes = request.required_bytes,
            effective_limit, min_chunk_bytes, work_units, "planning workspace chunks"
        );

        if effective_limit < min_chunk_bytes {
            return Err(cintxRsError::MemoryLimitExceeded {
                requested: request.required_bytes,
                limit: effective_limit,
            });
        }

        let max_units_per_chunk = max_work_units_for_limit(request, effective_limit);
        let requested_units_per_chunk = if let Some(override_units) = self.chunk_size_override {
            cmp::min(override_units.max(1), work_units)
        } else {
            max_units_per_chunk
        };
        let units_per_chunk = cmp::min(requested_units_per_chunk, max_units_per_chunk);

        let chunk_count = work_units.div_ceil(units_per_chunk);
        let fallback_reason = if chunk_count > 1 {
            Some(
                if self.chunk_size_override.is_some()
                    && requested_units_per_chunk == units_per_chunk
                {
                    "chunk_size_override"
                } else {
                    "memory_limit"
                },
            )
        } else {
            None
        };

        let mut chunks = Vec::with_capacity(chunk_count);
        for index in 0..chunk_count {
            let start = index * units_per_chunk;
            let end = cmp::min(start + units_per_chunk, work_units);
            let bytes = bytes_for_range(request.required_bytes, work_units, start, end)
                .max(min_chunk_bytes);
            if bytes > effective_limit {
                return Err(cintxRsError::MemoryLimitExceeded {
                    requested: bytes,
                    limit: effective_limit,
                });
            }
            chunks.push(ChunkInfo {
                index,
                work_unit_start: start,
                work_unit_count: end - start,
                bytes,
            });
        }

        debug!(
            chunk_count = chunks.len(),
            fallback_reason = fallback_reason.unwrap_or("none"),
            "workspace chunk plan complete"
        );

        Ok(ChunkPlan {
            chunks,
            fallback_reason,
        })
    }
}

fn bytes_for_range(total_bytes: usize, total_units: usize, start: usize, end: usize) -> usize {
    let prefix = total_bytes.saturating_mul(start) / total_units;
    let suffix = total_bytes.saturating_mul(end) / total_units;
    suffix.saturating_sub(prefix).max(1)
}

fn max_work_units_for_limit(request: &WorkspaceRequest, limit_bytes: usize) -> usize {
    let work_units = request.work_units.max(1);
    if request.required_bytes == 0 {
        return work_units;
    }

    let max_units =
        ((limit_bytes as u128) * (work_units as u128) / (request.required_bytes as u128)) as usize;

    cmp::min(max_units.max(1), work_units)
}

#[derive(Debug)]
pub struct FallibleBuffer<T> {
    storage: Vec<MaybeUninit<T>>,
    alignment: usize,
}

impl<T> FallibleBuffer<T> {
    pub fn try_uninit(len: usize, alignment: usize) -> Result<Self, cintxRsError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(len)
            .map_err(|_| cintxRsError::HostAllocationFailed { bytes })?;
        unsafe {
            storage.set_len(len);
        }
        Ok(Self { storage, alignment })
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

pub trait WorkspaceAllocator {
    fn try_alloc(
        &mut self,
        bytes: usize,
        alignment: usize,
    ) -> Result<FallibleBuffer<u8>, cintxRsError>;
    fn release(&mut self, buffer: FallibleBuffer<u8>);
}

#[derive(Debug, Default)]
pub struct HostWorkspaceAllocator {
    allocations: usize,
    peak_bytes: usize,
}

impl HostWorkspaceAllocator {
    pub fn allocations(&self) -> usize {
        self.allocations
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }
}

impl WorkspaceAllocator for HostWorkspaceAllocator {
    fn try_alloc(
        &mut self,
        bytes: usize,
        alignment: usize,
    ) -> Result<FallibleBuffer<u8>, cintxRsError> {
        let buffer = FallibleBuffer::try_uninit(bytes.max(1), alignment)?;
        self.allocations += 1;
        self.peak_bytes = self.peak_bytes.max(bytes);
        Ok(buffer)
    }

    fn release(&mut self, _buffer: FallibleBuffer<u8>) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_planner_splits_to_fit_limit() {
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            ..ExecutionOptions::default()
        };
        let planner = ChunkPlanner::from_options(&opts);
        let request = WorkspaceRequest {
            required_bytes: 768,
            alignment: DEFAULT_ALIGNMENT_BYTES,
            work_units: 12,
            min_chunk_bytes: 64,
        };

        let plan = planner.plan(&request).expect("chunk plan should fit");

        assert_eq!(plan.fallback_reason, Some("memory_limit"));
        assert!(plan.chunks.len() > 1);
        assert!(plan.chunks.iter().all(|chunk| chunk.bytes <= 192));
    }

    #[test]
    fn chunk_planner_reports_limit_exceeded_when_no_chunk_can_fit() {
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(32),
            ..ExecutionOptions::default()
        };
        let planner = ChunkPlanner::from_options(&opts);
        let request = WorkspaceRequest {
            required_bytes: 256,
            alignment: DEFAULT_ALIGNMENT_BYTES,
            work_units: 4,
            min_chunk_bytes: 64,
        };

        let err = planner.plan(&request).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::MemoryLimitExceeded {
                requested: 256,
                limit: 32,
            }
        ));
    }

    #[test]
    fn chunk_size_override_is_clamped_to_the_memory_limit() {
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            chunk_size_override: Some(4),
            ..ExecutionOptions::default()
        };
        let planner = ChunkPlanner::from_options(&opts);
        let request = WorkspaceRequest {
            required_bytes: 768,
            alignment: DEFAULT_ALIGNMENT_BYTES,
            work_units: 12,
            min_chunk_bytes: 64,
        };

        let plan = planner.plan(&request).expect("chunk plan should fit");

        assert_eq!(plan.fallback_reason, Some("memory_limit"));
        assert_eq!(plan.chunks.len(), 4);
        assert!(plan.chunks.iter().all(|chunk| chunk.work_unit_count == 3));
        assert!(plan.chunks.iter().all(|chunk| chunk.bytes <= 192));
    }

    #[test]
    fn chunk_size_override_is_used_when_it_fits_the_memory_limit() {
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(320),
            chunk_size_override: Some(2),
            ..ExecutionOptions::default()
        };
        let planner = ChunkPlanner::from_options(&opts);
        let request = WorkspaceRequest {
            required_bytes: 768,
            alignment: DEFAULT_ALIGNMENT_BYTES,
            work_units: 12,
            min_chunk_bytes: 64,
        };

        let plan = planner.plan(&request).expect("chunk plan should fit");

        assert_eq!(plan.fallback_reason, Some("chunk_size_override"));
        assert_eq!(plan.chunks.len(), 6);
        assert!(plan.chunks.iter().all(|chunk| chunk.work_unit_count == 2));
        assert!(plan.chunks.iter().all(|chunk| chunk.bytes <= 320));
    }

    #[test]
    fn planning_matches_checks_backend_contract() {
        use crate::options::{BackendCapabilityToken, BackendIntent, BackendKind};

        // Baseline: all contract fields match - should return true.
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_intent: BackendIntent {
                backend: BackendKind::Wgpu,
                selector: "auto".to_owned(),
            },
            backend_capability_token: BackendCapabilityToken {
                adapter_name: "test-adapter".to_owned(),
                backend_api: "wgpu".to_owned(),
                capability_fingerprint: 42,
            },
            ..ExecutionOptions::default()
        };
        let query = WorkspaceQuery {
            bytes: 192,
            alignment: DEFAULT_ALIGNMENT_BYTES,
            required_bytes: 192,
            chunk_count: 1,
            work_units: 4,
            min_chunk_bytes: 64,
            fallback_reason: None,
            chunks: vec![],
            memory_limit_bytes: opts.memory_limit_bytes,
            chunk_size_override: opts.chunk_size_override,
            backend_intent: opts.backend_intent.clone(),
            backend_capability_token: opts.backend_capability_token.clone(),
        };
        assert!(query.planning_matches(&opts), "matching contract should return true");

        // Change backend_intent.backend — should return false.
        let mut opts_different_backend = opts.clone();
        opts_different_backend.backend_intent.backend = BackendKind::Cpu;
        assert!(
            !query.planning_matches(&opts_different_backend),
            "backend kind drift must fail planning_matches"
        );

        // Change backend_intent.selector — should return false.
        let mut opts_different_selector = opts.clone();
        opts_different_selector.backend_intent.selector = "device:1".to_owned();
        assert!(
            !query.planning_matches(&opts_different_selector),
            "selector drift must fail planning_matches"
        );

        // Change capability_fingerprint — should return false.
        let mut opts_different_token = opts.clone();
        opts_different_token.backend_capability_token.capability_fingerprint = 99;
        assert!(
            !query.planning_matches(&opts_different_token),
            "capability token drift must fail planning_matches"
        );
    }
}
