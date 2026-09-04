//! `def2_speed_memory_optimization_plan.md` S6 — the batched Schwarz builder.
//!
//! # The gap this closes
//!
//! `cintx_driver::build_schwarz_table` takes a `DiagonalEvaluator` the caller
//! supplies, and the throughput benchmark supplies vendored libcint — which is
//! the right choice *there*, because building the table from one engine keeps
//! the screened work list identical for both and stops screening from
//! advantaging either side.
//!
//! It is not a choice a production caller has. Without a batched builder they
//! would evaluate `nbas^2 / 2` diagonal quartets through the per-tuple path, at
//! roughly the per-launch cost the whole batch surface exists to avoid, and the
//! list they screened would not be the list the benchmark measured.
//!
//! So the gate is agreement with the vendor-built table, element for element,
//! and then the property that matters downstream: **the two tables screen the
//! same work list**. A bound that differs in the last bits is harmless; a bound
//! that keeps a different set of quartets is not.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{ResidentTwoEBasis, TwoEBatchOptions, schwarz_bounds};
use cintx_driver::{
    BasisView, DiagonalEvaluator, DriverError, ShellPair, build_schwarz_table, enumerate_pairs,
    enumerate_quartets, screen_quartets,
};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, sulfur_dioxide, water};

struct VendorDiagonal<'a> {
    arrays: &'a RawArrays,
}

impl DiagonalEvaluator for VendorDiagonal<'_> {
    fn eval_diagonal(&mut self, pair: ShellPair, out: &mut [f64]) -> Result<(), DriverError> {
        vendor_ffi::vendor_int2e_sph(
            out,
            &[pair.i as i32, pair.j as i32, pair.i as i32, pair.j as i32],
            &self.arrays.atm,
            self.arrays.natm() as i32,
            &self.arrays.bas,
            self.arrays.nbas() as i32,
            &self.arrays.env,
        );
        Ok(())
    }
}

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// Build both tables for `molecule` and compare them, then compare what they
/// screen.
fn compare(label: &str, molecule: &Molecule, tolerance: f64) {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let pairs = enumerate_pairs(&view);
    let quartets = enumerate_quartets(&pairs);

    let mut vendor = VendorDiagonal { arrays: &arrays };
    let reference = build_schwarz_table(&view, &pairs, &mut vendor).expect("vendor table");

    let shells = batch_shells(&arrays);
    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");
    let bounds = schwarz_bounds(&backend, &resident, TwoEBatchOptions::default())
        .expect("batched schwarz table");

    let nbas = view.nbas();
    let mut worst = 0.0_f64;
    let mut worst_pair = (0_usize, 0_usize);
    for pair in &pairs {
        let mine = bounds[pair.i as usize * nbas + pair.j as usize];
        let theirs = reference.get(*pair);
        let diff = (mine - theirs).abs();
        if diff > worst {
            worst = diff;
            worst_pair = (pair.i as usize, pair.j as usize);
        }
        // The table is symmetric by construction, because `(ij|ij)` is.
        assert_eq!(
            bounds[pair.j as usize * nbas + pair.i as usize],
            mine,
            "{label}: table must be symmetric at ({}, {})",
            pair.i,
            pair.j
        );
    }

    // A relative gate: `Q` is a square root of an integral, so its scale is the
    // integral's, and an absolute tolerance would be meaningless across a table
    // whose entries span orders of magnitude.
    let scale = reference.max().max(f64::MIN_POSITIVE);
    println!(
        "{label}: {} pairs, worst |Q_cintx - Q_vendor| = {worst:.3e} at {worst_pair:?}, \
         table max {scale:.3e}, relative {:.3e}",
        pairs.len(),
        worst / scale
    );
    assert!(
        worst / scale < 1e-12,
        "{label}: Schwarz bounds differ from the vendor by {:.3e} of table scale",
        worst / scale
    );

    // What actually matters downstream: both tables keep the same quartets.
    let (vendor_kept, vendor_report) = screen_quartets(&quartets, &reference, tolerance);
    let mine = cintx_driver::SchwarzTable::from_square_matrix(&bounds, nbas);
    let (my_kept, my_report) = screen_quartets(&quartets, &mine, tolerance);
    assert_eq!(
        my_report.kept, vendor_report.kept,
        "{label}: the two tables must screen to the same size"
    );
    assert_eq!(
        my_kept, vendor_kept,
        "{label}: the two tables must keep the same quartets"
    );
    println!(
        "  screened at {tolerance:.0e}: {} of {} kept by both",
        my_report.kept, my_report.total
    );
}

#[test]
fn batched_schwarz_matches_the_vendor_built_table() {
    compare("H2O / def2-SVP", &water(StandardBasis::Def2Svp), 1e-10);
    compare(
        "SO2 / def2-SVP",
        &sulfur_dioxide(StandardBasis::Def2Svp),
        1e-10,
    );
    // A def2-TZVP diagonal quartet `(f d | f d)` needs `nroots = 6`, so this
    // case exists only where the extended device path is compiled in. That is
    // the same envelope every TZVP batch lives under, not a property of the
    // Schwarz builder.
    #[cfg(feature = "extended-device-rys")]
    compare("H2O / def2-TZVP", &water(StandardBasis::Def2Tzvp), 1e-10);
}
