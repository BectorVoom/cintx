use crate::workspace::{ChunkInfo, WorkspaceQuery};

/// Planner-owned deterministic chunk schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkSchedule {
    chunks: Vec<ChunkInfo>,
}

impl ChunkSchedule {
    pub fn chunks(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    pub fn total_work_units(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.work_unit_count).sum()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Return chunks in a deterministic order independent of backend behavior.
pub fn schedule_chunks(query: &WorkspaceQuery) -> ChunkSchedule {
    let mut chunks = query.chunks.clone();
    chunks.sort_by_key(|chunk| (chunk.work_unit_start, chunk.index));
    ChunkSchedule { chunks }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_chunks_orders_by_work_unit_start() {
        let query = WorkspaceQuery {
            bytes: 128,
            alignment: 64,
            required_bytes: 128,
            chunk_count: 2,
            work_units: 4,
            min_chunk_bytes: 32,
            fallback_reason: Some("memory_limit"),
            chunks: vec![
                ChunkInfo {
                    index: 1,
                    work_unit_start: 2,
                    work_unit_count: 2,
                    bytes: 64,
                },
                ChunkInfo {
                    index: 0,
                    work_unit_start: 0,
                    work_unit_count: 2,
                    bytes: 64,
                },
            ],
            memory_limit_bytes: Some(64),
            chunk_size_override: None,
            backend_intent: crate::options::BackendIntent::default(),
            backend_capability_token: crate::options::BackendCapabilityToken::default(),
        };

        let schedule = schedule_chunks(&query);
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule.total_work_units(), 4);
        assert_eq!(schedule.chunks()[0].index, 0);
        assert_eq!(schedule.chunks()[1].index, 1);
    }
}
