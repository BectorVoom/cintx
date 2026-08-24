//! Bucket execution and auditable statistics.

use crate::basis_view::BasisView;
use crate::bucket::{Bucket, LaunchTier};
use crate::error::DriverError;
use crate::worklist::ShellQuartet;
use std::time::{Duration, Instant};

/// Evaluates one shell quartet's spherical AO block into `out`.
///
/// Implemented once per engine (cintx, reference libcint) so the same work-list
/// can drive both.
pub trait QuartetEvaluator {
    /// # Errors
    /// Returns [`DriverError`] if evaluation fails.
    fn eval_quartet(&mut self, quartet: ShellQuartet, out: &mut [f64]) -> Result<(), DriverError>;

    /// Human-readable engine name, used in benchmark reporting.
    fn engine_name(&self) -> &'static str;
}

/// Statistics that make a claimed speedup auditable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchStats {
    pub buckets: usize,
    pub quartets_evaluated: usize,
    pub quartets_failed: usize,
    pub integrals_written: usize,
    /// Per-tier quartet counts, in [`LaunchTier`] declaration order.
    pub tier_counts: [usize; 3],
    pub elapsed: Duration,
}

impl BatchStats {
    /// Integrals per second, or `None` when nothing was timed.
    #[must_use]
    pub fn integrals_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return None;
        }
        Some(self.integrals_written as f64 / seconds)
    }

    /// Mean wall-clock per quartet.
    #[must_use]
    pub fn per_quartet(&self) -> Option<Duration> {
        if self.quartets_evaluated == 0 {
            return None;
        }
        Some(self.elapsed / self.quartets_evaluated as u32)
    }
}

/// Concatenated AO blocks plus the offset table that locates each quartet.
#[derive(Clone, Debug, Default)]
pub struct BatchOutput {
    pub values: Vec<f64>,
    /// `offsets[n]` is where quartet `n`'s block starts in `values`.
    pub offsets: Vec<usize>,
    pub quartets: Vec<ShellQuartet>,
    pub stats: BatchStats,
}

fn tier_index(tier: LaunchTier) -> usize {
    match tier {
        LaunchTier::ThreadPerQuartet => 0,
        LaunchTier::CubePerQuartetShared => 1,
        LaunchTier::CubePerQuartetGlobal => 2,
    }
}

/// Run every bucket through `evaluator`, collecting blocks and statistics.
///
/// `shared_memory_bytes` only affects tier *accounting*; it does not change
/// which evaluator runs, so the reported tier histogram describes the work
/// regardless of which engine executed it.
///
/// A quartet whose evaluation fails is counted in
/// [`BatchStats::quartets_failed`] and its block is left zeroed rather than
/// aborting the run — a partially-supported envelope should produce a coverage
/// report, not a bare panic.
///
/// # Errors
/// Returns [`DriverError`] only for allocation-shaped failures; per-quartet
/// evaluation failures are tallied, not propagated.
pub fn run_buckets<E: QuartetEvaluator>(
    basis: &BasisView<'_>,
    buckets: &[Bucket],
    evaluator: &mut E,
    shared_memory_bytes: usize,
) -> Result<BatchOutput, DriverError> {
    let mut output = BatchOutput::default();
    output.stats.buckets = buckets.len();

    // Size the output buffer up front so timing measures evaluation, not
    // reallocation.
    let mut total = 0_usize;
    for bucket in buckets {
        for &quartet in &bucket.quartets {
            total += block_len(basis, quartet);
        }
    }
    output.values.resize(total, 0.0);
    output.offsets.reserve(total.min(1 << 20));

    let start = Instant::now();
    let mut cursor = 0_usize;
    for bucket in buckets {
        let tier = bucket.class.tier(shared_memory_bytes);
        output.stats.tier_counts[tier_index(tier)] += bucket.quartets.len();

        for &quartet in &bucket.quartets {
            let len = block_len(basis, quartet);
            let slice = &mut output.values[cursor..cursor + len];
            match evaluator.eval_quartet(quartet, slice) {
                Ok(()) => output.stats.quartets_evaluated += 1,
                Err(_) => {
                    output.stats.quartets_failed += 1;
                    slice.fill(0.0);
                }
            }
            output.offsets.push(cursor);
            output.quartets.push(quartet);
            cursor += len;
        }
    }
    output.stats.elapsed = start.elapsed();
    output.stats.integrals_written = cursor;
    Ok(output)
}

/// Spherical AO block length for a quartet.
#[must_use]
pub fn block_len(basis: &BasisView<'_>, quartet: ShellQuartet) -> usize {
    basis.nsph(quartet.i as usize)
        * basis.nsph(quartet.j as usize)
        * basis.nsph(quartet.k as usize)
        * basis.nsph(quartet.l as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_view::BAS_SLOTS;
    use crate::bucket::bucket_quartets;
    use crate::worklist::{enumerate_pairs, enumerate_quartets};

    /// Minimal synthetic basis: `nbas` shells with the given angular momenta.
    fn synthetic(l: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut bas = vec![0_i32; l.len() * BAS_SLOTS];
        for (shell, &value) in l.iter().enumerate() {
            bas[shell * BAS_SLOTS + crate::basis_view::ANG_OF] = value;
            bas[shell * BAS_SLOTS + crate::basis_view::NPRIM_OF] = 1;
            bas[shell * BAS_SLOTS + crate::basis_view::NCTR_OF] = 1;
        }
        (
            vec![0_i32; crate::basis_view::ATM_SLOTS],
            bas,
            vec![0.0; 32],
        )
    }

    struct CountingEvaluator {
        calls: usize,
    }

    impl QuartetEvaluator for CountingEvaluator {
        fn eval_quartet(
            &mut self,
            _quartet: ShellQuartet,
            out: &mut [f64],
        ) -> Result<(), DriverError> {
            self.calls += 1;
            out.fill(1.0);
            Ok(())
        }
        fn engine_name(&self) -> &'static str {
            "counting"
        }
    }

    /// Every enumerated quartet must be evaluated exactly once, and the output
    /// buffer must be exactly filled — no gaps, no overlap.
    #[test]
    fn every_quartet_is_evaluated_once_and_output_is_dense() {
        let (atm, bas, env) = synthetic(&[0, 1, 2]);
        let basis = BasisView::new(&atm, &bas, &env);
        let pairs = enumerate_pairs(&basis);
        let quartets = enumerate_quartets(&pairs);
        let buckets = bucket_quartets(&basis, &quartets);

        let mut evaluator = CountingEvaluator { calls: 0 };
        let output = run_buckets(&basis, &buckets, &mut evaluator, 48 * 1024).unwrap();

        assert_eq!(evaluator.calls, quartets.len());
        assert_eq!(output.stats.quartets_evaluated, quartets.len());
        assert_eq!(output.stats.quartets_failed, 0);
        assert_eq!(output.offsets.len(), quartets.len());
        assert!(
            output.values.iter().all(|&v| v == 1.0),
            "output must be densely written with no gaps"
        );
        assert_eq!(output.stats.integrals_written, output.values.len());
    }

    /// Bucketing must partition the quartet list exactly.
    #[test]
    fn buckets_partition_the_quartet_list() {
        let (atm, bas, env) = synthetic(&[0, 0, 1, 1, 2, 3]);
        let basis = BasisView::new(&atm, &bas, &env);
        let quartets = enumerate_quartets(&enumerate_pairs(&basis));
        let buckets = bucket_quartets(&basis, &quartets);

        let bucketed: usize = buckets.iter().map(Bucket::len).sum();
        assert_eq!(bucketed, quartets.len());

        let mut seen: Vec<ShellQuartet> = buckets
            .iter()
            .flat_map(|b| b.quartets.iter().copied())
            .collect();
        seen.sort_by_key(|q| (q.i, q.j, q.k, q.l));
        let mut expected = quartets.clone();
        expected.sort_by_key(|q| (q.i, q.j, q.k, q.l));
        assert_eq!(seen, expected);
    }

    /// A failing evaluator must be tallied, not propagated, so a partial
    /// envelope yields a coverage report.
    #[test]
    fn evaluation_failures_are_tallied_not_propagated() {
        struct AlwaysFails;
        impl QuartetEvaluator for AlwaysFails {
            fn eval_quartet(
                &mut self,
                quartet: ShellQuartet,
                _out: &mut [f64],
            ) -> Result<(), DriverError> {
                Err(DriverError::Evaluation {
                    shells: quartet.shls(),
                    detail: "unsupported".to_owned(),
                })
            }
            fn engine_name(&self) -> &'static str {
                "always-fails"
            }
        }

        let (atm, bas, env) = synthetic(&[0, 1]);
        let basis = BasisView::new(&atm, &bas, &env);
        let quartets = enumerate_quartets(&enumerate_pairs(&basis));
        let buckets = bucket_quartets(&basis, &quartets);
        let output = run_buckets(&basis, &buckets, &mut AlwaysFails, 48 * 1024).unwrap();

        assert_eq!(output.stats.quartets_failed, quartets.len());
        assert_eq!(output.stats.quartets_evaluated, 0);
        assert!(output.values.iter().all(|&v| v == 0.0));
    }
}
