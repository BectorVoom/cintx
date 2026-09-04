//! `def2_speed_memory_optimization_plan.md` S1 — libcint's `expcutoff` primitive-pair screen.
//!
//! # What changed, and what the gate has to be
//!
//! The batched 2e kernel used to walk all `nprim^4` primitive quartets of a
//! shell quartet, with the bra pair as the outer loop. It now walks a resident
//! primitive-pair table with the **ket pair outer**, which is libcint's own
//! nesting (`cint2e.c:192-230`), and applies the vendor's three `expcutoff`
//! tests.
//!
//! Two consequences, and the second is why this file exists rather than a
//! one-line bit-identity assertion:
//!
//! 1. Some primitive quartets are no longer evaluated. They are exactly the ones
//!    the *vendor* does not evaluate either, so this moves cintx towards
//!    libcint, not away from it.
//! 2. The order of the contraction sum changed, because the loop nesting did.
//!    Floating-point addition is not associative, so results differ from the
//!    old kernel in the last bits **even with the cutoff disabled**.
//!
//! So "bit-identical to the previous kernel" is not available and would be the
//! wrong thing to ask for: the previous order was not the vendor's. What is
//! asked for instead:
//!
//! - **Unscreened is a real superset.** With the cutoff at `+inf` every
//!   primitive pair survives, so the dispatched primitive-quartet count equals
//!   the raw `Σ nprim^4`. That is what makes it the A/B reference.
//! - **Screening does not cost vendor agreement.** The screened run's
//!   element-wise agreement with vendored libcint is no worse than the
//!   unscreened run's, on every fixture, at the project's `1e-12`.
//! - **Screening does real work.** On a fixture with diffuse, well-separated
//!   primitives it drops a measurable fraction rather than being a no-op that
//!   happens to pass.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    LIBCINT_EXPCUTOFF, PairTable, PairTableOptions, ResidentTwoEBasis,
    evaluate_2e_quartet_batch_resident,
};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, methane, sulfur_dioxide, water};

/// Project oracle tolerance.
const TOL: f64 = 1e-12;

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// Every canonical quartet of `molecule`, unscreened by Schwarz.
///
/// Schwarz screening is a separate, orthogonal decision; mixing it in here
/// would let a Schwarz-dropped quartet hide a pair-cutoff disagreement.
fn work_list(arrays: &RawArrays) -> Vec<[u32; 4]> {
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let pairs = enumerate_pairs(&basis);
    enumerate_quartets(&pairs)
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect()
}

/// Vendored libcint over the same list, concatenated in list order.
fn vendor_values(arrays: &RawArrays, list: &[[u32; 4]]) -> Vec<f64> {
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let mut out = Vec::new();
    let mut block = Vec::new();
    for quartet in list {
        let len = quartet
            .iter()
            .map(|&s| basis.nsph(s as usize))
            .product::<usize>();
        block.clear();
        block.resize(len, 0.0);
        vendor_ffi::vendor_int2e_sph(
            &mut block,
            &[
                quartet[0] as i32,
                quartet[1] as i32,
                quartet[2] as i32,
                quartet[3] as i32,
            ],
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );
        out.extend_from_slice(&block);
    }
    out
}

/// Largest absolute difference between `values` and the vendor's.
fn max_abs_diff(values: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(values.len(), reference.len(), "element counts must agree");
    values
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// One fixture, both cutoff settings, compared against the vendor.
struct CutoffComparison {
    label: &'static str,
    quartets: usize,
    primitive_quartets_unscreened: u64,
    primitive_quartets_screened: u64,
    raw_primitive_quartets: u64,
    diff_unscreened: f64,
    diff_screened: f64,
    /// Largest gap between the two cintx runs themselves.
    diff_between_settings: f64,
}

fn compare(label: &'static str, molecule: &Molecule) -> CutoffComparison {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let list = work_list(&arrays);
    let backend = cpu_backend();
    let reference = vendor_values(&arrays, &list);

    let unscreened_basis =
        ResidentTwoEBasis::new_with(&backend, &shells, PairTableOptions::unscreened())
            .expect("unscreened residency");
    let unscreened = evaluate_2e_quartet_batch_resident(&backend, &unscreened_basis, &list)
        .expect("unscreened batch");

    let screened_basis =
        ResidentTwoEBasis::new_with(&backend, &shells, PairTableOptions::default())
            .expect("screened residency");
    let screened = evaluate_2e_quartet_batch_resident(&backend, &screened_basis, &list)
        .expect("screened batch");

    // `Σ nprim_i·nprim_j·nprim_k·nprim_l` over the list, computed here rather
    // than read from the stats, so the stats are checked against an independent
    // count rather than against themselves.
    let raw: u64 = list
        .iter()
        .map(|q| {
            q.iter()
                .map(|&s| u64::from(shells[s as usize].nprim))
                .product::<u64>()
        })
        .sum();

    CutoffComparison {
        label,
        quartets: list.len(),
        primitive_quartets_unscreened: unscreened.stats.primitive_quartets_evaluated,
        primitive_quartets_screened: screened.stats.primitive_quartets_evaluated,
        raw_primitive_quartets: raw,
        diff_unscreened: max_abs_diff(&unscreened.values, &reference),
        diff_screened: max_abs_diff(&screened.values, &reference),
        diff_between_settings: max_abs_diff(&unscreened.values, &screened.values),
    }
}

/// The `+inf` setting must keep every primitive pair — otherwise it is not the
/// A/B reference it is used as everywhere below.
#[test]
fn unscreened_evaluates_every_primitive_quartet() {
    // Pure table construction, so def2-TZVP is admissible here whatever the
    // device ceiling is: no quartet is dispatched.
    for (label, molecule) in [
        ("H2O/def2-SVP", water(StandardBasis::Def2Svp)),
        ("H2O/def2-TZVP", water(StandardBasis::Def2Tzvp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let shells = batch_shells(&arrays);
        let table = PairTable::build(&shells, PairTableOptions::unscreened());
        assert_eq!(table.pairs_dropped, 0, "{label}: unscreened dropped a pair");
        for (i, bra) in shells.iter().enumerate() {
            for (j, ket) in shells.iter().enumerate() {
                assert_eq!(
                    table.pair_count(i as u32, j as u32),
                    bra.nprim * ket.nprim,
                    "{label}: pair ({i},{j})"
                );
            }
        }
    }
}

/// The gate. Screening must not cost vendor agreement, on any fixture.
#[test]
fn screened_matches_the_vendor_at_least_as_well_as_unscreened() {
    let cases = [
        compare("H2O / def2-SVP", &water(StandardBasis::Def2Svp)),
        compare("CH4 / def2-SVP", &methane(StandardBasis::Def2Svp)),
        compare("SO2 / def2-SVP", &sulfur_dioxide(StandardBasis::Def2Svp)),
        // def2-TZVP reaches `nroots = 6`, so it is a case only where the
        // extended device path is compiled in — the same envelope every TZVP
        // batch lives under.
        #[cfg(feature = "extended-device-rys")]
        compare("H2O / def2-TZVP", &water(StandardBasis::Def2Tzvp)),
    ];

    println!(
        "\n{:<18} {:>8} {:>14} {:>14} {:>7} {:>11} {:>11} {:>11}",
        "case",
        "quartets",
        "prim (raw)",
        "prim (screened)",
        "skip%",
        "diff unscr",
        "diff scr",
        "scr-unscr"
    );
    for case in &cases {
        println!(
            "{:<18} {:>8} {:>14} {:>14} {:>6.1}% {:>11.3e} {:>11.3e} {:>11.3e}",
            case.label,
            case.quartets,
            case.raw_primitive_quartets,
            case.primitive_quartets_screened,
            100.0
                * (1.0
                    - case.primitive_quartets_screened as f64 / case.raw_primitive_quartets as f64),
            case.diff_unscreened,
            case.diff_screened,
            case.diff_between_settings,
        );
    }

    for case in &cases {
        // The unscreened setting dispatches exactly the raw count.
        assert_eq!(
            case.primitive_quartets_unscreened, case.raw_primitive_quartets,
            "{}: unscreened must dispatch every primitive quartet",
            case.label
        );
        // Both settings agree with the vendor at the project tolerance.
        assert!(
            case.diff_unscreened < TOL,
            "{}: unscreened diff {:.3e} exceeds {TOL:.0e}",
            case.label,
            case.diff_unscreened
        );
        assert!(
            case.diff_screened < TOL,
            "{}: screened diff {:.3e} exceeds {TOL:.0e}",
            case.label,
            case.diff_screened
        );
        // The point of S1: dropping the vendor's own dropped terms does not
        // move cintx away from the vendor. A small slack absorbs the last-bit
        // reordering the skip itself causes; it is three orders below the gate.
        assert!(
            case.diff_screened <= case.diff_unscreened.max(1e-15) * 4.0,
            "{}: screening worsened vendor agreement, {:.3e} -> {:.3e}",
            case.label,
            case.diff_unscreened,
            case.diff_screened
        );
    }
}

/// Screening must actually screen. A fixture whose primitives are diffuse and
/// well separated is where libcint's estimate earns its keep, and SO2/def2-SVP
/// is the benchmark fixture that shows it most clearly.
#[test]
fn the_cutoff_drops_real_work_on_a_second_row_fixture() {
    let case = compare("SO2 / def2-SVP", &sulfur_dioxide(StandardBasis::Def2Svp));
    assert!(
        case.primitive_quartets_screened < case.raw_primitive_quartets,
        "the cutoff dropped nothing on SO2/def2-SVP"
    );
    let dropped =
        1.0 - case.primitive_quartets_screened as f64 / case.raw_primitive_quartets as f64;
    assert!(
        dropped > 0.05,
        "expected the cutoff to drop more than 5% of SO2/def2-SVP primitive quartets, got {:.1}%",
        dropped * 100.0
    );
}

/// The default threshold is the vendor's, not a number of cintx's choosing.
///
/// `EXPCUTOFF` is `60` in `src/cint_config.h.in:27`, and `g2e.c:57` selects it
/// whenever `env[PTR_EXPCUTOFF]` is zero — which is what `cintx-basis` emits.
#[test]
fn the_default_threshold_is_libcints() {
    assert_eq!(LIBCINT_EXPCUTOFF, 60.0);
    assert_eq!(PairTableOptions::default().expcutoff, LIBCINT_EXPCUTOFF);
    assert!(PairTableOptions::unscreened().is_unscreened());
}
