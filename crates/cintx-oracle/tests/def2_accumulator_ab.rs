//! `def2_speed_memory_optimization_plan.md` S2 — private accumulation, A/B'd in one process.
//!
//! # Why this is a test and not a benchmark run
//!
//! Absolute times on the development host vary by up to 2x between processes
//! for identical work: vendored libcint's own H2O/def2-SVP figure ranged from
//! 2.3 ms to 4.9 ms across one afternoon. Running the two accumulator settings
//! in separate processes and comparing them measures the machine, not the
//! change — the first attempt at exactly that produced a 14% win on one
//! workload and a 41% loss on another, from the same pair of binaries.
//!
//! So the two settings are alternated **inside one process**, interleaved
//! repeat by repeat so a drifting machine perturbs both equally, and each is
//! reported by its best run. `set_accumulator_slots_max` exists for this.
//!
//! # What is being compared
//!
//! - **global**: the contraction sum lands in `cart_out` — a kernel argument,
//!   so a pointer the compiler cannot prove unaliased — once per primitive
//!   quartet per element.
//! - **private**: it lands in a per-work-item array of at most `ACC_SLOTS`
//!   elements and is written out once per quartet.
//!
//! Both must produce **bit-identical** values: the same primitive quartets are
//! summed into the same element in the same order, in a different place.

#![cfg(feature = "cpu")]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    ACC_SLOTS_DEFAULT, ResidentTwoEBasis, evaluate_2e_quartet_batch_resident, prewarm_2e_work_list,
    set_accumulator_slots_max,
};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, methane, sulfur_dioxide, water};

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

fn repeats() -> usize {
    std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(7)
}

struct Ab {
    label: &'static str,
    quartets: usize,
    global_ns: u64,
    private_ns: u64,
    identical: bool,
}

/// Run one workload under both settings, alternating, best of `repeats`.
fn ab(label: &'static str, molecule: &Molecule) -> Ab {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");

    // Both settings are the same compiled program — the ceiling is a runtime
    // scalar — so one prewarm covers both and neither pays a JIT cost the other
    // does not.
    prewarm_2e_work_list(&backend, &shells, &list).expect("prewarm");

    let mut global_ns = u64::MAX;
    let mut private_ns = u64::MAX;
    let mut global_values = Vec::new();
    let mut private_values = Vec::new();

    for _ in 0..repeats() {
        set_accumulator_slots_max(0);
        let start = std::time::Instant::now();
        let out = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("global");
        global_ns = global_ns.min(start.elapsed().as_nanos() as u64);
        global_values = out.values;

        set_accumulator_slots_max(ACC_SLOTS_DEFAULT as u32);
        let start = std::time::Instant::now();
        let out = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("private");
        private_ns = private_ns.min(start.elapsed().as_nanos() as u64);
        private_values = out.values;
    }
    set_accumulator_slots_max(ACC_SLOTS_DEFAULT as u32);

    let identical = global_values.len() == private_values.len()
        && global_values
            .iter()
            .zip(&private_values)
            .all(|(a, b)| a.to_bits() == b.to_bits());

    Ab {
        label,
        quartets: list.len(),
        global_ns,
        private_ns,
        identical,
    }
}

/// The two settings must agree bit for bit, and the comparison is printed so a
/// reader sees the size of the effect rather than only that it was positive.
#[test]
#[ignore = "timing comparison; run explicitly in release"]
fn private_accumulation_is_bit_identical_and_measured() {
    let cases = [
        ab("H2O / def2-SVP", &water(StandardBasis::Def2Svp)),
        ab("CH4 / def2-SVP", &methane(StandardBasis::Def2Svp)),
        ab("SO2 / def2-SVP", &sulfur_dioxide(StandardBasis::Def2Svp)),
    ];

    println!(
        "\n{:<18} {:>9} {:>13} {:>13} {:>9}  {}",
        "case", "quartets", "global (ms)", "private (ms)", "speedup", "identical"
    );
    for case in &cases {
        println!(
            "{:<18} {:>9} {:>13.3} {:>13.3} {:>8.3}x  {}",
            case.label,
            case.quartets,
            case.global_ns as f64 / 1e6,
            case.private_ns as f64 / 1e6,
            case.global_ns as f64 / case.private_ns as f64,
            case.identical
        );
    }

    for case in &cases {
        assert!(
            case.identical,
            "{}: the accumulator must not change a single bit",
            case.label
        );
    }
}

/// The bit-identity half, cheap enough for the default suite.
#[test]
fn the_accumulator_changes_no_value() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");

    set_accumulator_slots_max(0);
    let global = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("global");
    set_accumulator_slots_max(ACC_SLOTS_DEFAULT as u32);
    let private = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("private");

    assert_eq!(global.values.len(), private.values.len());
    for (index, (a, b)) in global.values.iter().zip(&private.values).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {index}: global {a:.17e} vs private {b:.17e}"
        );
    }
}
