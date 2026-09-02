//! `def2_speed_precision_plan.md` D2.1 — the specialization prewarm moves the
//! first batch's compilation out of the measurement.
//!
//! # Why this test is alone in its own binary
//!
//! CubeCL caches compiled programs per device, and that cache is
//! **process-global**: a second `ResolvedBackend::from_intent` hands back a
//! client that already knows every program the first one built. Measured here
//! directly — an H2O/def2-TZVP prewarm on a "fresh" backend costs 6.7 s, and
//! the SO2/def2-TZVP prewarm that follows it in the same process costs 0.11 s,
//! because def2-TZVP's launch signatures are a property of the `l` values
//! present and both molecules present `l = 0..3`.
//!
//! So a cold start is observable exactly once per process, and any test sharing
//! a binary with this one would either warm the cache first or be warmed by it.
//! Cargo gives each `tests/*.rs` its own binary; this file uses that, and holds
//! exactly one test, on purpose.
//!
//! # The trap the measurement is here to catch
//!
//! A backend specializes a program per `(nroots, ibase, kbase, per_unit,
//! cube_dim)`, and `cube_dim` is part of that identity. On the per-unit (CPU)
//! decomposition it is `min(parallel_units, n_quartets, memory_cap)`, so a
//! one-quartet warm-up compiles a **one-lane** program while the real batch
//! compiles its own — the warm-up costs time and buys nothing.
//! `prewarm_2e_quartet_classes` saturates its item count past `parallel_units`
//! for exactly that reason, and a regression there would show up here as the
//! first post-prewarm batch still paying full compilation.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{evaluate_2e_quartet_batch, prewarm_2e_work_list};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, water};

fn backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// The def2-SVP water quartet list, unscreened.
fn svp_work_list() -> (Vec<cintx_cubecl::BatchShell>, Vec<[u32; 4]>) {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    let list = quartets
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    (batch_shells(&arrays), list)
}

/// **D2.1.** A prewarm must move the compilation, not merely repeat it.
///
/// The measurement is a ratio, not a wall-clock threshold: two batches after a
/// prewarm, and the claim is that the *first* one is no longer the expensive
/// one. A cold first batch on this fixture is dominated by compilation and runs
/// many times the steady-state cost, so a factor of 2 is a wide margin around a
/// large effect rather than a tight bound on a small one — the point is to
/// catch a prewarm that warmed the wrong specialization, which shows up as the
/// first batch still paying full compilation.
#[test]
fn prewarm_removes_the_first_batch_compilation() {
    let (shells, list) = svp_work_list();
    let backend = backend();

    // The prewarm *is* the cold start: it is the first thing in this process to
    // touch these specializations, so its own wall time is the compilation the
    // caller would otherwise have paid inside their first batch.
    let report = prewarm_2e_work_list(&backend, &shells, &list).expect("prewarm");
    let first_start = std::time::Instant::now();
    evaluate_2e_quartet_batch(&backend, &shells, &list).expect("first batch after prewarm");
    let first_after_prewarm = first_start.elapsed();
    let second_start = std::time::Instant::now();
    evaluate_2e_quartet_batch(&backend, &shells, &list).expect("second batch after prewarm");
    let second_after_prewarm = second_start.elapsed();

    println!(
        "prewarm: {} classes -> {} signatures, {} launches, {} items/class, {:.3} s \
         ({:.1} ms/signature)",
        report.classes,
        report.signatures,
        report.launches,
        report.items_per_class,
        report.elapsed.as_secs_f64(),
        report.ms_per_signature(),
    );
    println!(
        "  prewarmed:  first batch {:.4} s, second {:.4} s  (first/second = {:.1}x)",
        first_after_prewarm.as_secs_f64(),
        second_after_prewarm.as_secs_f64(),
        first_after_prewarm.as_secs_f64() / second_after_prewarm.as_secs_f64().max(f64::MIN_POSITIVE),
    );

    assert!(
        report.refused.is_empty(),
        "def2-SVP has no class past the device ceiling, so the prewarm must refuse \
         nothing; got {:?}",
        report.refused
    );
    assert!(
        report.signatures > 0 && report.launches >= report.signatures,
        "a prewarm that issued no launch compiled nothing: {report:?}"
    );

    // The precondition: the prewarm must actually have compiled something. If
    // this process had somehow arrived warm, the assertion below would pass for
    // the wrong reason.
    let steady = second_after_prewarm.as_secs_f64().max(f64::MIN_POSITIVE);
    let compile_ratio = report.elapsed.as_secs_f64() / steady;
    assert!(
        compile_ratio > 2.0,
        "precondition: the prewarm cost only {compile_ratio:.1}x a steady-state \
         batch, so it compiled little or nothing — this process did not start cold \
         and the measurement below means nothing"
    );

    let prewarmed_ratio = first_after_prewarm.as_secs_f64() / steady;
    assert!(
        prewarmed_ratio < 2.0,
        "after a prewarm costing {compile_ratio:.1}x a steady batch, the first real \
         batch still cost {prewarmed_ratio:.1}x one — the prewarm compiled a \
         different specialization from the one the batch asks for"
    );
}

