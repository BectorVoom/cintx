//! `def2_speed_memory_optimization_plan.md` M1 — bounded-memory batch execution.
//!
//! # What a memory budget has to mean
//!
//! `ExecutionOptions::memory_limit_bytes` has existed since the per-tuple
//! planner and was silently ignored by the batched 2e path: a whole-molecule
//! work list allocated its spherical output, every group's Cartesian buffer and
//! every group's scratch, whatever the caller had asked for. M1 makes the budget
//! real, and this file pins the three properties that make it trustworthy.
//!
//! 1. **Chunking changes no arithmetic.** A chunk is a range of quartets; each
//!    quartet's evaluation is self-contained and the transform writes rather
//!    than accumulates, so a run split across many dispatches must be
//!    **bit-identical** to the same run in one. Not "within tolerance" —
//!    identical, because nothing was reordered.
//! 2. **A refusal is clean.** A budget too small to hold even the chunked shape
//!    fails with `MemoryLimitExceeded` before the output buffer is allocated and
//!    before any launch, so there is nothing partially written to observe.
//! 3. **The peak actually falls.** Interleaving the transform with the dispatch
//!    turns the Cartesian half of the host peak from a sum over groups into a
//!    maximum over them. That is the whole point, and it is measured rather than
//!    asserted from the shape of the code.

#![cfg(feature = "cpu")]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{StandardBasis, to_raw_arrays};
use cintx_core::cintxRsError;
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    ResidentTwoEBasis, TwoEBatchOptions, evaluate_2e_quartet_batch_into,
    evaluate_2e_quartet_batch_resident, evaluate_2e_quartet_batch_with,
};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, sulfur_dioxide, water};

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

fn options(limit: Option<usize>) -> TwoEBatchOptions {
    TwoEBatchOptions {
        primitive_tolerance: 0.0,
        memory_limit_bytes: limit,
        expcutoff: None,
    }
}

/// A budget that forces many dispatches must reproduce the unbounded run
/// **exactly**. This is the gate that makes chunking safe to turn on.
#[test]
fn chunked_evaluation_is_bit_identical_to_unchunked() {
    // def2-SVP rather than def2-TZVP: TZVP's `(pf|ff)` classes need `nroots = 6`
    // and so the `extended-device-rys` feature, and a memory property should be
    // gated in every build rather than only the feature-complete one.
    let arrays = to_raw_arrays(&sulfur_dioxide(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&basis))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");

    let unbounded =
        evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("unbounded batch");

    println!(
        "unbounded: output {} B, largest group cart {} B, scratch peak {} B, {} dispatches",
        unbounded.stats.host_output_bytes,
        unbounded.stats.device_out_bytes_peak,
        unbounded.stats.device_g_slab_bytes_peak,
        unbounded.stats.kernel_launch_count,
    );
    // Ask for a peak that leaves room for only a quarter of the largest group's
    // Cartesian block, so the budget-fitting loop has to split groups to reach
    // it. Anything looser and the first plan fits, which is correct behaviour
    // but does not exercise chunking.
    // The output, the scratch slab and the quartet/shape tables are
    // irreducible — splitting a group does not shrink any of them — so a
    // reachable limit has to cover all three. What is left over is what the
    // budget can chunk, and half of the largest group's block forces the fitting
    // loop to split it. Table bytes are doubled because chunking adds a shape
    // row per class per new group.
    let limit = unbounded.stats.host_output_bytes
        + unbounded.stats.device_g_slab_bytes_peak
        + 2 * unbounded.stats.device_table_bytes_total
        + unbounded.stats.device_out_bytes_peak / 2;
    let bounded = evaluate_2e_quartet_batch_with(&backend, &resident, &list, options(Some(limit)))
        .expect("bounded batch");

    assert!(
        bounded.stats.kernel_launch_count > unbounded.stats.kernel_launch_count,
        "the budget must actually force extra dispatches: {} vs {}",
        bounded.stats.kernel_launch_count,
        unbounded.stats.kernel_launch_count
    );
    assert_eq!(
        bounded.offsets, unbounded.offsets,
        "output layout must match"
    );
    assert_eq!(
        bounded.values.len(),
        unbounded.values.len(),
        "element count must match"
    );
    // Bit-identical, element by element. `assert_eq!` on the whole vector would
    // say only that they differ; this says where.
    for (index, (a, b)) in bounded
        .values
        .iter()
        .zip(unbounded.values.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {index} differs: chunked {a:.17e} vs unchunked {b:.17e}"
        );
    }
    println!(
        "SO2/def2-SVP: {} quartets, {} dispatches bounded vs {} unbounded, values bit-identical",
        list.len(),
        bounded.stats.kernel_launch_count,
        unbounded.stats.kernel_launch_count
    );
}

/// A budget below what a single dispatch needs must be refused up front, with
/// the typed error and no allocation of the caller's output.
#[test]
fn an_impossible_budget_is_refused_before_any_launch() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&basis))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");

    // One kibibyte cannot hold this list's spherical output, let alone a
    // dispatch's scratch.
    let error = evaluate_2e_quartet_batch_with(&backend, &resident, &list, options(Some(1024)))
        .expect_err("a 1 KiB budget must be refused");
    match error {
        cintxRsError::MemoryLimitExceeded { requested, limit } => {
            assert_eq!(limit, 1024);
            assert!(
                requested > limit,
                "the refusal must report what was needed: {requested} vs {limit}"
            );
        }
        other => panic!("expected MemoryLimitExceeded, got {other:?}"),
    }
}

/// A refusal must leave a caller-owned buffer completely untouched.
///
/// `evaluate_2e_quartet_batch_into` writes real values straight into the
/// caller's own buffer as chunks complete. If the budget is only checked
/// per-chunk as each one dispatches, a work list that plans into more than
/// one chunk can have earlier chunks' real values already written before a
/// later chunk's plan is found to exceed the limit — leaving the caller with
/// a buffer that is part real data, part whatever it held before the call.
/// The budget must instead be validated for every chunk before any of them
/// writes anything.
#[test]
fn a_refused_budget_leaves_the_callers_buffer_untouched() {
    let arrays = to_raw_arrays(&sulfur_dioxide(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&basis))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");
    let unbounded =
        evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("unbounded batch");

    // One kibibyte cannot hold even one dispatch's scratch, let alone the
    // list's spherical output — refused regardless of how many chunks the
    // budget would otherwise split the list into.
    let sentinel = -1.0_f64;
    let mut values = vec![sentinel; unbounded.values.len()];
    let error = evaluate_2e_quartet_batch_into(
        &backend,
        &resident,
        &list,
        options(Some(1024)),
        &mut values,
    )
    .expect_err("a 1 KiB budget must be refused");
    assert!(matches!(error, cintxRsError::MemoryLimitExceeded { .. }));
    assert!(
        values.iter().all(|&value| value == sentinel),
        "a refused call must not have written any chunk's values into the caller's buffer"
    );
}

/// An empty work list is not a memory problem, whatever the budget.
#[test]
fn an_empty_work_list_is_admitted_under_any_budget() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");
    let output = evaluate_2e_quartet_batch_with(&backend, &resident, &[], options(Some(1)))
        .expect("an empty list allocates nothing");
    assert!(output.values.is_empty());
}

/// The budget buys a bounded Cartesian intermediate, and the benchmark's
/// heaviest workload is where that is worth having.
///
/// SO2/def2-TZVP materializes a 172.8 MiB Cartesian intermediate against a
/// 99.8 MiB spherical output. Unbounded, the host holds both; under a budget it
/// holds the output and one chunk. Both points are printed, because the cost of
/// the second is real and belongs beside the saving.
#[test]
#[cfg(feature = "extended-device-rys")]
#[ignore = "allocates ~100 MiB and runs the whole SO2/def2-TZVP list; run explicitly"]
fn a_budget_bounds_the_cartesian_intermediate() {
    let arrays = to_raw_arrays(&sulfur_dioxide(StandardBasis::Def2Tzvp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&basis))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");
    let unbounded =
        evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("unbounded batch");

    // Ask for the output plus 40 MiB of working set.
    let limit = unbounded.stats.host_output_bytes + 40 * 1024 * 1024;
    let bounded = evaluate_2e_quartet_batch_with(&backend, &resident, &list, options(Some(limit)))
        .expect("bounded batch");

    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let ratio = |stats: &cintx_cubecl::TwoEBatchStats| {
        (stats.host_output_bytes + stats.host_cart_bytes_peak) as f64
            / stats.host_output_bytes as f64
    };
    println!(
        "SO2/def2-TZVP output {:.1} MiB\n  unbounded: cart {:.1} MiB, peak {:.2}x, {} chunks, {} dispatches\n  bounded:   cart {:.1} MiB, peak {:.2}x, {} chunks, {} dispatches",
        mib(unbounded.stats.host_output_bytes),
        mib(unbounded.stats.host_cart_bytes_peak),
        ratio(&unbounded.stats),
        unbounded.stats.chunk_count,
        unbounded.stats.kernel_launch_count,
        mib(bounded.stats.host_cart_bytes_peak),
        ratio(&bounded.stats),
        bounded.stats.chunk_count,
        bounded.stats.kernel_launch_count,
    );

    assert_eq!(unbounded.stats.chunk_count, 1, "the default is one chunk");
    assert!(
        bounded.stats.chunk_count > 1,
        "a 40 MiB working set must split a 172.8 MiB intermediate"
    );
    assert!(
        bounded.stats.host_cart_bytes_peak < unbounded.stats.host_cart_bytes_peak / 3,
        "the budget must cut the Cartesian peak substantially: {} vs {}",
        bounded.stats.host_cart_bytes_peak,
        unbounded.stats.host_cart_bytes_peak
    );
    assert!(
        ratio(&bounded.stats) <= 1.5,
        "bounded peak ratio {:.2}x should be near the output size",
        ratio(&bounded.stats)
    );
    // And it is still the same answer.
    for (index, (a, b)) in bounded
        .values
        .iter()
        .zip(unbounded.values.iter())
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index} differs");
    }
}
