//! `def2_speed_precision_plan.md` D2.1 and D2.2 — the specialization prewarm,
//! and the claim that a def2 work list costs one launch per launch *signature*
//! rather than one per class or one per quartet.
//!
//! # D2.1: coverage and refusal reporting
//!
//! The *timing* claim — that a prewarm moves the first batch's compilation out
//! of the measurement — lives in `def2_prewarm_cold_start`, alone in its own
//! test binary, because CubeCL's compiled-program cache is process-global and a
//! cold start can only be observed once per process. What this file covers is
//! the other half: which classes a prewarm warms, and what it does with one it
//! cannot.
//!
//! # D2.2: consolidation
//!
//! Task 35-M1 merges every angular-momentum class sharing a launch signature
//! into one dispatch. [`def2_batches_launch_once_per_signature`] pins that to
//! the def2 work lists themselves — the number the plan's exit criterion names
//! — rather than to a synthetic list that happens to merge well.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{evaluate_2e_quartet_batch, prewarm_2e_quartet_classes};
use cintx_driver::{BasisView, bucket_quartets, enumerate_pairs, enumerate_quartets};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, sulfur_dioxide, water};
use std::collections::BTreeSet;

/// The CPU backend.
///
/// Deliberately not called "fresh": CubeCL's compiled-program cache is
/// process-global, so every call here hands back a client that already knows
/// whatever earlier tests in this binary compiled. Nothing in this file
/// measures time, so that costs nothing — but the name should not suggest a
/// cold start that is not available.
fn fresh_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// A prewarm reports, rather than fails on, a class the backend cannot serve.
///
/// Without `extended-device-rys` a def2-TZVP basis contains `nroots` 6-7
/// classes that every 2e entry point refuses. A warm-up that propagated that
/// refusal would turn an optional optimization into a new way for a program not
/// to start, so it records the classes and warms the rest.
#[cfg(not(feature = "extended-device-rys"))]
#[test]
fn prewarm_reports_refused_classes_instead_of_failing() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Tzvp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let report =
        prewarm_2e_quartet_classes(&fresh_backend(), &shells).expect("prewarm must not fail");
    println!(
        "def2-TZVP without the extended path: {} classes, {} refused, {} launches",
        report.classes,
        report.refused.len(),
        report.launches
    );
    assert!(
        !report.refused.is_empty(),
        "def2-TZVP has nroots 6-7 classes, which this build cannot serve; the \
         prewarm must say so"
    );
    assert!(
        report.launches > 0,
        "the classes the backend *can* serve must still be warmed"
    );
    for (class, reason) in &report.refused {
        let nroots = (class.iter().map(|&l| l as usize).sum::<usize>()) / 2 + 1;
        assert!(
            nroots > cintx_cubecl::BASE_DEVICE_NROOTS,
            "class {class:?} (nroots={nroots}) is inside the base envelope and must \
             not have been refused: {reason}"
        );
    }
}

/// With the extended path on, a def2-TZVP prewarm refuses nothing — the same
/// claim `def2_device_coverage` makes for the batch surfaces, made for the
/// warm-up path that precedes them.
#[cfg(feature = "extended-device-rys")]
#[test]
fn prewarm_covers_def2_tzvp_completely() {
    for (label, molecule) in [
        ("H2O", water(StandardBasis::Def2Tzvp)),
        ("SO2", sulfur_dioxide(StandardBasis::Def2Tzvp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let shells = batch_shells(&arrays);
        let report = prewarm_2e_quartet_classes(&fresh_backend(), &shells).expect("prewarm");
        println!(
            "{label}/def2-TZVP prewarm: {} classes -> {} signatures, {} launches, \
             {:.2} s ({:.0} ms/signature)",
            report.classes,
            report.signatures,
            report.launches,
            report.elapsed.as_secs_f64(),
            report.ms_per_signature(),
        );
        assert!(
            report.refused.is_empty(),
            "{label}/def2-TZVP prewarm refused {:?}",
            report.refused
        );
    }
}

/// **D2.2.** One dispatch per launch signature, on the def2 work lists.
///
/// The plan's exit criterion is "one launch per (class, chunk)" verified via
/// `ExecutionStats.kernel_launch_count`, with any per-quartet residue closed.
/// What the batch path actually achieves is stronger — one launch per
/// *signature*, with every angular-momentum class sharing `(nroots, ibase,
/// kbase)` merged into it — so that is what this asserts, against the signature
/// count derived independently from the bucket list.
#[test]
fn def2_batches_launch_once_per_signature() {
    for (label, molecule) in [
        ("H2O/def2-SVP", water(StandardBasis::Def2Svp)),
        ("SO2/def2-SVP", sulfur_dioxide(StandardBasis::Def2Svp)),
        #[cfg(feature = "extended-device-rys")]
        ("H2O/def2-TZVP", water(StandardBasis::Def2Tzvp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
        let quartets = enumerate_quartets(&enumerate_pairs(&basis));
        let list: Vec<[u32; 4]> = quartets
            .iter()
            .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
            .collect();
        let shells = batch_shells(&arrays);

        // The signature count, derived from the work list without asking the
        // backend: `(ibase, kbase, nroots)` from the canonical class, exactly as
        // `TwoELaunchSignature::of` computes it from `build_2e_shape`.
        let mut signatures: BTreeSet<(bool, bool, u32)> = BTreeSet::new();
        for bucket in bucket_quartets(&basis, &quartets) {
            let [li, lj, lk, ll] = bucket.class.angular_momenta;
            signatures.insert((li > lj, lk > ll, bucket.class.nroots));
        }

        let output = evaluate_2e_quartet_batch(&fresh_backend(), &shells, &list).expect("batch");
        println!(
            "{label:<14} quartets={} classes={} signatures(derived)={} \
             launches={} launch_classes={} readbacks={}",
            list.len(),
            bucket_quartets(&basis, &quartets).len(),
            signatures.len(),
            output.stats.kernel_launch_count,
            output.stats.launch_classes,
            output.stats.readback_count,
        );

        assert_eq!(
            output.stats.kernel_launch_count,
            signatures.len(),
            "{label}: one dispatch per launch signature, no per-quartet residue"
        );
        assert_eq!(
            output.stats.readback_count, output.stats.kernel_launch_count,
            "{label}: one readback per dispatch"
        );
        assert!(
            output.stats.launch_classes >= output.stats.kernel_launch_count,
            "{label}: merging can only reduce launches below the class count, never \
             raise it above"
        );
    }
}
