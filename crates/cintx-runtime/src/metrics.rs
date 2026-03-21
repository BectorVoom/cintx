use crate::workspace::{ChunkInfo, WorkspaceQuery};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionStats {
    pub workspace_bytes: usize,
    pub required_workspace_bytes: usize,
    pub peak_workspace_bytes: usize,
    pub chunk_count: usize,
    pub planned_batches: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
    pub fallback_reason: Option<&'static str>,
}

impl ExecutionStats {
    pub fn empty(workspace: &WorkspaceQuery) -> Self {
        Self {
            workspace_bytes: workspace.bytes,
            required_workspace_bytes: workspace.required_bytes,
            peak_workspace_bytes: 0,
            chunk_count: workspace.chunks.len(),
            planned_batches: workspace.chunks.iter().map(|chunk| chunk.work_unit_count).sum(),
            transfer_bytes: 0,
            not0: 0,
            fallback_reason: workspace.fallback_reason,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunMetrics {
    pub chunk_count: usize,
    pub peak_workspace_bytes: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
}

impl RunMetrics {
    pub fn observe_chunk(&mut self, chunk: &ChunkInfo, workspace_bytes: usize) {
        let _ = chunk;
        self.chunk_count = self.chunk_count.saturating_add(1);
        self.peak_workspace_bytes = self.peak_workspace_bytes.max(workspace_bytes);
    }

    pub fn observe_transfer_bytes(&mut self, bytes: usize) {
        self.transfer_bytes = self.transfer_bytes.saturating_add(bytes);
    }

    pub fn observe_not0(&mut self, not0: i32) {
        self.not0 = self.not0.saturating_add(not0.max(0));
    }

    pub fn merge_backend_stats(&mut self, stats: &ExecutionStats) {
        self.peak_workspace_bytes = self.peak_workspace_bytes.max(stats.peak_workspace_bytes);
        self.observe_transfer_bytes(stats.transfer_bytes);
        self.observe_not0(stats.not0);
    }

    pub fn finish(self, workspace: &WorkspaceQuery) -> ExecutionStats {
        ExecutionStats {
            workspace_bytes: workspace.bytes,
            required_workspace_bytes: workspace.required_bytes,
            peak_workspace_bytes: self.peak_workspace_bytes,
            chunk_count: self.chunk_count.max(workspace.chunks.len()),
            planned_batches: workspace.chunks.iter().map(|chunk| chunk.work_unit_count).sum(),
            transfer_bytes: self.transfer_bytes,
            not0: self.not0,
            fallback_reason: workspace.fallback_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_metrics_collect_transfer_and_not0() {
        let mut metrics = RunMetrics::default();
        metrics.observe_transfer_bytes(64);
        metrics.observe_transfer_bytes(32);
        metrics.observe_not0(4);
        metrics.observe_not0(3);

        assert_eq!(metrics.transfer_bytes, 96);
        assert_eq!(metrics.not0, 7);
    }
}
