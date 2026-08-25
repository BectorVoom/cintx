//! Running a batch's host cart-to-sph transform across threads (Task 36-T2).
//!
//! # Why this is safe to parallelise, and why the gate stays bit-identical
//!
//! Every batched family ends with the same loop: for each tuple, for each
//! contraction block, for each component, transform one Cartesian block to
//! spherical and write it into the caller's AO grid. **Each output element is
//! produced by exactly one tuple.** There is no accumulation across tuples —
//! the transform writes, it does not add — so splitting the tuple list across
//! threads does not reorder a single floating-point summation. The result is
//! bit-identical to the serial loop by construction, not by tolerance.
//!
//! That claim is worth stating in words because a reader who sees `rayon` next
//! to a bit-identity gate will otherwise assume the gate was loosened. It was
//! not: `def2_2e_batch_parity`, `def2_pair_batch_parity`,
//! `def2_1e_deriv_batch_parity` and `def2_3c2e_deriv_batch_parity` all still
//! compare element by element against the per-tuple path.
//!
//! # How disjointness is established without `unsafe`
//!
//! A tuple's destination is `output.values[offsets[n] .. offsets[n] + len_n]`,
//! and `offsets` is built as a running total in the caller's tuple order — so
//! the blocks are contiguous, non-overlapping and *in order*. Repeated
//! [`slice::split_at_mut`] in that order hands out one `&mut [f64]` per tuple
//! with no aliasing and no raw pointers. The transform then writes at offsets
//! relative to its own block.
//!
//! # Scratch
//!
//! Per-block buffers cannot be shared across threads, so each worker takes its
//! own through [`for_each_block`]'s `init` closure. `rayon` creates that state
//! once per worker and reuses it for every block that worker takes, so the
//! allocation Task 36-T1 removed does not come back per block.
//!
//! # Thread count
//!
//! `CINTX_HOST_TRANSFORM_THREADS` pins the worker count, mirroring the
//! `CINTX_2E_PER_UNIT` precedent:
//!
//! - **unset** — `rayon`'s global pool, i.e. one worker per core;
//! - **`1`** — the serial path, no pool and no work-stealing, which is the A/B
//!   baseline a speed-up is measured against;
//! - **`n > 1`** — a pool of exactly `n` workers, built once per process.
//!
//! Setting `CINTX_HOST_TRANSFORM_PROFILE` forces the serial path, because the
//! allocate/c2s/scatter attribution is a per-block wall-clock split and means
//! nothing summed across workers racing each other.

use std::sync::OnceLock;

use rayon::prelude::*;

/// Worker count for the host transform: `1` means run serially.
///
/// Read once per process — this is consulted per batch, and `std::env::var`
/// there would be charged to the very thing being measured.
pub fn host_transform_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        // A profiling run is a serial run; see the module docs.
        if super::profile::profiling_enabled() {
            return 1;
        }
        match std::env::var("CINTX_HOST_TRANSFORM_THREADS") {
            Ok(value) => value.trim().parse::<usize>().unwrap_or(0),
            Err(_) => 0,
        }
    })
}

/// Below this many tuples the transform runs serially whatever the thread
/// setting says.
///
/// Measured on the def2-SVP benchmarks (Task 36-T2), CPU backend, after Task
/// 36-T1 made the serial transform ~6x cheaper on the 3-index families:
///
/// | work list | tuples | serial | parallel |
/// |---|---:|---:|---:|
/// | `int2e` CH4/def2-SVP | 14 706 | 1.85 ms | **0.72 ms** |
/// | `int3c2e` RI-J def2/J | 1 950 | **0.18 ms** | 0.25 ms |
/// | `int3c2e_ip1` | 1 728 | **0.19 ms** | 0.32 ms |
///
/// Short lists lose to the fan-out: each of those transforms is now a fraction
/// of a millisecond in total, and splitting it across sixteen workers costs
/// more than it saves. A Fock build's real work lists are far above this;
/// the threshold only decides the regime where the answer was "don't bother".
///
/// `CINTX_HOST_TRANSFORM_MIN_JOBS` overrides it, for measuring either side.
fn min_parallel_jobs() -> usize {
    static MIN_JOBS: OnceLock<usize> = OnceLock::new();
    *MIN_JOBS.get_or_init(|| {
        std::env::var("CINTX_HOST_TRANSFORM_MIN_JOBS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4096)
    })
}

/// The pinned pool, built only when `CINTX_HOST_TRANSFORM_THREADS > 1`.
fn pinned_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = host_transform_threads();
        if threads <= 1 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("cintx-c2s-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// Run `body` over every job, in parallel unless pinned to one thread, and
/// return the per-worker states.
///
/// `init` builds one state per worker; `body` receives that state and one job.
/// Jobs must be independent — see the module docs for why the batch transform's
/// are. The states come back so a caller can fold whatever the workers
/// accumulated (the allocate/c2s/scatter profile, in practice) instead of
/// reaching for a shared counter.
pub fn for_each_block<J, S, I, B>(jobs: Vec<J>, init: I, body: B) -> Vec<S>
where
    J: Send,
    S: Send,
    I: Fn() -> S + Sync + Send,
    B: Fn(&mut S, J) + Sync + Send,
{
    if host_transform_threads() == 1 || jobs.len() < min_parallel_jobs().max(2) {
        let mut state = init();
        for job in jobs {
            body(&mut state, job);
        }
        return vec![state];
    }
    let run = || {
        jobs.into_par_iter()
            .fold(&init, |mut state, job| {
                body(&mut state, job);
                state
            })
            .collect::<Vec<S>>()
    };
    match pinned_pool() {
        Some(pool) => pool.install(run),
        None => run(),
    }
}

/// Split `values` into one disjoint block per tuple, in the caller's order.
///
/// `lens[n]` is tuple `n`'s block length; they must sum to `values.len()`,
/// which is exactly how the batch entry points size the output. Returns the
/// blocks in the same order, so `blocks[n]` is what tuple `n` writes.
///
/// This is what makes the parallel transform `unsafe`-free: the borrow checker
/// proves the blocks do not alias, rather than a comment asserting it.
pub fn split_output_blocks<'a>(values: &'a mut [f64], lens: &[usize]) -> Vec<&'a mut [f64]> {
    let mut blocks = Vec::with_capacity(lens.len());
    let mut rest = values;
    for &len in lens {
        let (block, tail) = rest.split_at_mut(len);
        blocks.push(block);
        rest = tail;
    }
    debug_assert!(rest.is_empty(), "block lengths must cover the whole output");
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocks a split hands out cover the buffer exactly once, in order.
    #[test]
    fn split_covers_the_output_in_order() {
        let mut values = vec![0.0_f64; 10];
        let lens = [3_usize, 0, 5, 2];
        {
            let blocks = split_output_blocks(&mut values, &lens);
            assert_eq!(blocks.len(), lens.len());
            for (index, block) in blocks.into_iter().enumerate() {
                assert_eq!(block.len(), lens[index]);
                for slot in block.iter_mut() {
                    *slot = index as f64 + 1.0;
                }
            }
        }
        assert_eq!(
            values,
            vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0]
        );
    }

    /// Every job runs exactly once, whatever the worker count.
    #[test]
    fn for_each_block_visits_every_job() {
        // Long enough to clear `min_parallel_jobs` on a default build, so the
        // parallel arm is the one exercised.
        let count = min_parallel_jobs() * 2;
        let jobs: Vec<usize> = (0..count).collect();
        let states = for_each_block(jobs, Vec::new, |seen: &mut Vec<usize>, job| seen.push(job));
        let mut seen: Vec<usize> = states.into_iter().flatten().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..count).collect::<Vec<_>>());
    }

    /// A short list takes the serial path regardless of the thread setting —
    /// spinning up work-stealing for a handful of blocks costs more than it
    /// saves, which is what [`min_parallel_jobs`] records.
    #[test]
    fn single_job_runs_without_a_pool() {
        let states = for_each_block(vec![7_usize], || 0_usize, |state, job| *state += job);
        assert_eq!(states, vec![7]);
    }
}
