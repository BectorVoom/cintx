//! The launch-geometry autotuner must change speed and nothing else.
//!
//! Phase 6 of `docs/design/cubecl_speed_optimization_plan.md` states the exit
//! gate plainly: *tuning never changes results*. The 2e batch kernel covers its
//! quartet list grid-stride and splits each quartet's contraction block across
//! the cube's lanes, so that property holds by construction — but "by
//! construction" is exactly the kind of claim that stops being true after an
//! unrelated kernel edit, so it is pinned here.
//!
//! The test uses the kernel as its own reference. A work list long enough to
//! cross `cintx_cubecl::tuning::MIN_TUNE_ITEMS` goes through the tuner, which
//! benchmarks several cube widths and keeps the fastest. The same quartets
//! evaluated in chunks *below* that floor never reach the tuner at all and run
//! on the heuristic geometry. Both must produce the same bits.
//!
//! Tuning is off by default, so each parity test turns it on explicitly rather
//! than depending on the ambient policy — otherwise the comparison would be
//! between two heuristic runs and would prove nothing.
//!
//! Run with:
//! `cargo test -p cintx-cubecl --features cpu --test tuned_geometry_parity`

#![cfg(feature = "cpu")]

use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::kernels::one_electron::{BatchAtom, OneEOperator, evaluate_1e_pair_batch};
use cintx_cubecl::kernels::two_electron::{BatchShell, evaluate_2e_quartet_batch};
use cintx_cubecl::tuning::{AutotunePolicy, MIN_TUNE_ITEMS, policy, set_policy, tuned_key_count};
use cintx_runtime::{BackendIntent, BackendKind};
use std::sync::{Mutex, OnceLock};

/// Serializes the parity tests.
///
/// `tuned_key_count` is a process-global counter, so a test that reads it
/// before and after its own dispatches would otherwise see keys admitted by a
/// sibling test running on another thread. The tuning policy these tests set is
/// process-global too.
fn tuning_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

/// Chunk size for the reference path: strictly below the tuning floor, so those
/// dispatches are guaranteed to run on the heuristic geometry.
const UNTUNED_CHUNK: usize = MIN_TUNE_ITEMS / 4;

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("cpu backend resolves")
}

/// Four contracted shells on distinct centers: one `s`, one `p`, and two more
/// `s` shells with different exponents, so the quartet list is not degenerate.
fn shells() -> Vec<BatchShell> {
    let centers = [
        [0.0_f64, 0.0, 0.0],
        [0.0_f64, 0.0, 1.4],
        [0.9_f64, 0.0, -0.3],
        [-0.4_f64, 1.1, 0.2],
    ];
    let angular = [0_u8, 0, 1, 0];
    centers
        .iter()
        .zip(angular)
        .enumerate()
        .map(|(index, (center, l))| {
            let scale = 1.0 + 0.35 * index as f64;
            BatchShell {
                l,
                nprim: 3,
                nctr: 1,
                exponents: vec![3.1 * scale, 0.9 * scale, 0.31 * scale],
                coefficients: vec![0.21, 0.47, 0.36],
                center: *center,
            }
        })
        .collect()
}

/// Two point charges for the nuclear arm.
fn atoms() -> Vec<BatchAtom> {
    vec![
        BatchAtom {
            charge: 8.0,
            center: [0.0, 0.0, 0.0],
        },
        BatchAtom {
            charge: 1.0,
            center: [0.0, 0.0, 1.4],
        },
    ]
}

/// Every ordered shell pair, repeated until the list crosses the tuning floor.
fn pairs(n_shells: u32, repeats: usize) -> Vec<[u32; 2]> {
    let mut list = Vec::new();
    for _ in 0..repeats {
        for i in 0..n_shells {
            for j in 0..n_shells {
                list.push([i, j]);
            }
        }
    }
    list
}

/// Every quartet over the shell list — 256 of them, well past the tuning floor.
fn quartets(n_shells: u32) -> Vec<[u32; 4]> {
    let mut list = Vec::new();
    for i in 0..n_shells {
        for j in 0..n_shells {
            for k in 0..n_shells {
                for l in 0..n_shells {
                    list.push([i, j, k, l]);
                }
            }
        }
    }
    list
}

#[test]
fn tuned_and_untuned_dispatches_agree_bit_for_bit() {
    let _guard = tuning_guard()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_policy(AutotunePolicy::Balanced);
    let backend = cpu_backend();
    let shells = shells();
    let quartets = quartets(shells.len() as u32);
    assert!(
        quartets.len() > MIN_TUNE_ITEMS,
        "the work list must cross the tuning floor for this test to exercise the tuner"
    );

    // One batch: long enough that each launch group reaches the tuner.
    let tuned = evaluate_2e_quartet_batch(&backend, &shells, &quartets).expect("tuned batch runs");

    // The same quartets in sub-floor chunks: never tuned, heuristic geometry.
    let mut reference = Vec::with_capacity(tuned.values.len());
    for chunk in quartets.chunks(UNTUNED_CHUNK) {
        let out = evaluate_2e_quartet_batch(&backend, &shells, chunk).expect("chunked batch runs");
        reference.extend_from_slice(&out.values);
    }

    assert_eq!(
        tuned.values.len(),
        reference.len(),
        "tuning must not change how much output a work list produces"
    );
    for (index, (tuned_value, reference_value)) in
        tuned.values.iter().zip(reference.iter()).enumerate()
    {
        assert_eq!(
            tuned_value.to_bits(),
            reference_value.to_bits(),
            "element {index} differs between the tuned and untuned geometries: \
             {tuned_value} vs {reference_value}"
        );
    }

    if policy().enabled() {
        assert!(
            tuned_key_count() > 0,
            "the tuned path never ran; the parity check above proved nothing"
        );
    }
}

#[test]
fn a_second_run_reuses_the_tuned_geometry_and_agrees_with_the_first() {
    let _guard = tuning_guard()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_policy(AutotunePolicy::Balanced);
    let backend = cpu_backend();
    let shells = shells();
    let quartets = quartets(shells.len() as u32);

    let first = evaluate_2e_quartet_batch(&backend, &shells, &quartets).expect("first batch runs");
    let keys_after_first = tuned_key_count();
    let second =
        evaluate_2e_quartet_batch(&backend, &shells, &quartets).expect("second batch runs");

    assert_eq!(
        tuned_key_count(),
        keys_after_first,
        "a repeated dispatch must hit the tuning cache, not admit a new key"
    );
    for (index, (a, b)) in first.values.iter().zip(second.values.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {index} differs between the tuning run and the cached run"
        );
    }
}

#[test]
fn tuned_and_untuned_1e_dispatches_agree_bit_for_bit() {
    let _guard = tuning_guard()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_policy(AutotunePolicy::Balanced);
    let backend = cpu_backend();
    let shells = shells();
    let atoms = atoms();
    // 16 ordered pairs per repeat; 16 repeats puts every launch class past the
    // tuning floor.
    let pairs = pairs(shells.len() as u32, 16);
    assert!(pairs.len() > MIN_TUNE_ITEMS);

    for operator in [
        OneEOperator::Overlap,
        OneEOperator::Kinetic,
        OneEOperator::Nuclear,
    ] {
        let tuned = evaluate_1e_pair_batch(&backend, operator, &shells, &atoms, &pairs)
            .expect("tuned 1e batch runs");

        let mut reference = Vec::with_capacity(tuned.values.len());
        for chunk in pairs.chunks(UNTUNED_CHUNK) {
            let out = evaluate_1e_pair_batch(&backend, operator, &shells, &atoms, chunk)
                .expect("chunked 1e batch runs");
            reference.extend_from_slice(&out.values);
        }

        assert_eq!(
            tuned.values.len(),
            reference.len(),
            "{operator:?}: tuning must not change how much output a work list produces"
        );
        for (index, (tuned_value, reference_value)) in
            tuned.values.iter().zip(reference.iter()).enumerate()
        {
            assert_eq!(
                tuned_value.to_bits(),
                reference_value.to_bits(),
                "{operator:?}: element {index} differs between the tuned and untuned geometries: \
                 {tuned_value} vs {reference_value}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Throughput harness for the tuned geometry, in the spirit of the Task 34-A0
// `CINTX_2E_CUBE_DIM` A/B in `two_electron.rs`.
//
// The policy is process-global, so the comparison is between two runs:
//
//   CINTX_AUTOTUNE=off cargo test --release -p cintx-cubecl --features cpu \
//     --test tuned_geometry_parity -- --ignored --nocapture
//   cargo test --release -p cintx-cubecl --features cpu \
//     --test tuned_geometry_parity -- --ignored --nocapture
//
// The second run pays the tuning pass once and reads the cache afterwards, so
// run it twice to separate cold from warm. Unlike the parity tests above, this
// one honours the ambient policy rather than forcing it — comparing the two is
// the whole point of it.
// ─────────────────────────────────────────────────────────────────────────────

/// A wider shell set — `s`, `p` and `d` on eight centers — so the work list
/// spans several launch classes rather than one.
fn wide_shells() -> Vec<BatchShell> {
    let angular = [0_u8, 0, 1, 0, 1, 2, 0, 1];
    angular
        .iter()
        .enumerate()
        .map(|(index, &l)| {
            let scale = 1.0 + 0.23 * index as f64;
            let angle = 0.7 * index as f64;
            BatchShell {
                l,
                nprim: 3,
                nctr: 1,
                exponents: vec![4.2 * scale, 1.1 * scale, 0.29 * scale],
                coefficients: vec![0.18, 0.51, 0.34],
                center: [angle.cos() * 1.3, angle.sin() * 1.3, 0.2 * index as f64],
            }
        })
        .collect()
}

#[test]
#[ignore = "throughput measurement; run explicitly with --ignored --nocapture"]
fn tuned_vs_heuristic_throughput() {
    const REPEATS: usize = 25;

    let backend = cpu_backend();
    let shells = wide_shells();
    let quartets = quartets(shells.len() as u32);

    // Warm up: JIT, specialization, and (when enabled) the tuning pass itself.
    // Reported separately because that is exactly the cold-start cost the
    // persistent cache exists to remove on the *next* process.
    let warm = std::time::Instant::now();
    let _ = evaluate_2e_quartet_batch(&backend, &shells, &quartets).expect("warmup runs");
    let first_call = warm.elapsed();

    // Per-repeat samples, reported as a median rather than a mean: this host
    // shows run-to-run spreads of several hundred percent on a *fixed*
    // configuration, and one stalled repeat moves a mean by more than any
    // geometry does.
    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        let _ = evaluate_2e_quartet_batch(&backend, &shells, &quartets).expect("timed run");
        samples.push(start.elapsed());
    }
    samples.sort_unstable();

    println!(
        "policy={:?} quartets={} keys={} first_call={:?} min={:?} median={:?} max={:?} \
         ({:.2} us/quartet at the median)",
        policy(),
        quartets.len(),
        tuned_key_count(),
        first_call,
        samples[0],
        samples[REPEATS / 2],
        samples[REPEATS - 1],
        samples[REPEATS / 2].as_secs_f64() * 1e6 / quartets.len() as f64,
    );
}
