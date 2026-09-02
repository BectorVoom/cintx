//! `def2_speed_precision_plan.md` D0.3 + the executable form of gates G1 and G2.
//!
//! # The question this answers
//!
//! "Does def2-TZVP run on the device?" was, until this file, answered by
//! `def2_tzvp_exceeds_the_device_envelope`, which proves the *opposite* — that
//! TZVP has classes above `nroots = 5` — and by six separate extended-Rys parity
//! gates, each of which proves that *its* family gets the right answer where it
//! runs. Neither says that a whole def2 work list is served without a refusal.
//!
//! This file asks every batch surface, for every launch class a def2 work list
//! contains, whether it accepts the class. It does not compare values: the
//! parity gates already own that, and folding correctness in here would make a
//! coverage regression and an arithmetic regression look the same.
//!
//! # Why one representative quartet per class
//!
//! Device eligibility is a property of the launch class — the angular-momentum
//! tuple and the Rys order it implies — not of the particular shells. A class
//! is accepted or refused as a whole, in one `if nroots > ceiling` at the top of
//! a launcher. So a representative per class is a complete test of coverage and
//! costs seconds instead of hours; the SO2/def2-TZVP 2e list alone is ~200 k
//! quartets.
//!
//! # Fixtures
//!
//! H2O is the plan's reference workload. SO2 is the second-row molecule D0.1
//! asks for: sulfur carries `d` and `f` shells in def2-TZVP, so the heavy
//! classes are loaded by an atom with real contraction depth rather than by
//! oxygen alone.
//!
//! # The artifact
//!
//! `cintx_def2_coverage.json`, under the mandatory locations, carries the
//! per-basis, per-family class census and the accepted/refused split. D1's
//! coverage progress and D5's release matrix both read it, and G2's claim is
//! only auditable if the numbers behind it travel with it.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::raw::{ATOM_OF, BAS_SLOTS};
use cintx_basis::{Molecule, StandardBasis, to_raw_arrays, to_raw_arrays_with_auxiliary};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{RysFamily, device_nroots_ceiling, fma_fusion_verified};
use cintx_cubecl::{
    OneEDerivOperator, OneEOperator, ThreeC2eDerivFamily, evaluate_1e_deriv_pair_batch,
    evaluate_1e_pair_batch, evaluate_2e_quartet_batch, evaluate_3c2e_deriv_triple_batch,
    evaluate_3c2e_triple_batch,
};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_atoms, batch_shells, shell_l, sulfur_dioxide, water};
use serde_json::json;
use std::collections::BTreeMap;

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

// ─────────────────────────────────────────────────────────────────────────────
// Class census
// ─────────────────────────────────────────────────────────────────────────────

/// One family's coverage over one work list.
#[derive(Debug, Default)]
struct FamilyCoverage {
    /// Distinct angular-momentum classes in the list.
    classes: usize,
    /// Classes whose Rys order is above `BASE_DEVICE_NROOTS`.
    classes_above_base: usize,
    /// Classes the batch surface accepted.
    accepted: usize,
    /// Classes it refused, with the first refusal's message per class.
    refusals: Vec<(String, String)>,
    /// Rys orders present, ascending.
    orders: Vec<usize>,
}

impl FamilyCoverage {
    fn to_json(&self, family: &str, ceiling: usize) -> serde_json::Value {
        json!({
            "family": family,
            "device_nroots_ceiling": ceiling,
            "classes": self.classes,
            "classes_above_base_ceiling": self.classes_above_base,
            "accepted": self.accepted,
            "refused": self.refusals.len(),
            "nroots_present": self.orders,
            "refusals": self.refusals
                .iter()
                .map(|(class, detail)| json!({"class": class, "detail": detail}))
                .collect::<Vec<_>>(),
        })
    }
}

/// Walk one representative tuple per angular-momentum class and record whether
/// the surface accepted it.
///
/// `key_of` names the class; `nroots_of` is the family's own Rys-order formula,
/// which differs per family (the derivative shapes add a unit of angular
/// momentum) and so cannot be derived here.
fn census<T: Copy, K: Ord + std::fmt::Debug>(
    tuples: impl Iterator<Item = T>,
    key_of: impl Fn(T) -> K,
    nroots_of: impl Fn(T) -> usize,
    mut evaluate: impl FnMut(T) -> Result<(), String>,
) -> FamilyCoverage {
    let mut representatives: BTreeMap<K, T> = BTreeMap::new();
    for tuple in tuples {
        representatives.entry(key_of(tuple)).or_insert(tuple);
    }

    let mut coverage = FamilyCoverage {
        classes: representatives.len(),
        ..Default::default()
    };
    let mut orders = std::collections::BTreeSet::new();
    for (key, tuple) in representatives {
        let nroots = nroots_of(tuple);
        orders.insert(nroots);
        if nroots > cintx_cubecl::BASE_DEVICE_NROOTS {
            coverage.classes_above_base += 1;
        }
        match evaluate(tuple) {
            Ok(()) => coverage.accepted += 1,
            Err(detail) => coverage.refusals.push((format!("{key:?}"), detail)),
        }
    }
    coverage.orders = orders.into_iter().collect();
    coverage
}

/// Every family's coverage over one molecule/basis pair, as a JSON object.
fn coverage_for(label: &str, molecule: &Molecule, aux: StandardBasis) -> serde_json::Value {
    let backend = cpu_backend();
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let atoms = batch_atoms(&arrays);
    let nbas = arrays.nbas();
    let l = |s: usize| shell_l(&arrays, s);

    // ── 2e ────────────────────────────────────────────────────────────────
    let quartets = (0..nbas).flat_map(move |i| {
        (0..=i).flat_map(move |j| (0..=i).flat_map(move |k| (0..=k).map(move |m| [i, j, k, m])))
    });
    let two_e = census(
        quartets,
        |q| [l(q[0]), l(q[1]), l(q[2]), l(q[3])],
        |q| (l(q[0]) + l(q[1]) + l(q[2]) + l(q[3])) / 2 + 1,
        |q| {
            let list = [[q[0] as u32, q[1] as u32, q[2] as u32, q[3] as u32]];
            evaluate_2e_quartet_batch(&backend, &shells, &list)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );

    // ── 1e scalar and 1e gradient ─────────────────────────────────────────
    let pairs: Vec<[usize; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i, j]))
        .collect();
    let one_e = census(
        pairs.iter().copied(),
        |p| [l(p[0]), l(p[1])],
        |p| (l(p[0]) + l(p[1])) / 2 + 1,
        |p| {
            let list = [[p[0] as u32, p[1] as u32]];
            evaluate_1e_pair_batch(&backend, OneEOperator::Nuclear, &shells, &atoms, &list)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );
    let one_e_deriv = census(
        pairs.iter().copied(),
        |p| [l(p[0]), l(p[1])],
        |p| (l(p[0]) + l(p[1])).div_ceil(2) + 1,
        |p| {
            let list = [[p[0] as u32, p[1] as u32]];
            evaluate_1e_deriv_pair_batch(&backend, OneEDerivOperator::IpNuc, &shells, &atoms, &list)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );

    // ── 3c2e and its derivatives, against the RI-J auxiliary set ──────────
    let aux_arrays = to_raw_arrays_with_auxiliary(molecule, aux).expect("combined arrays");
    let aux_shells = batch_shells(&aux_arrays);
    let la = |s: usize| shell_l(&aux_arrays, s);
    let orbital = aux_arrays.orbital_shells();
    let auxiliary = aux_arrays.auxiliary_shells();
    let triples: Vec<[usize; 3]> = orbital
        .clone()
        .flat_map(|mu| {
            let auxiliary = auxiliary.clone();
            (mu..orbital.end).flat_map(move |nu| auxiliary.clone().map(move |p| [mu, nu, p]))
        })
        .collect();

    let three_c2e = census(
        triples.iter().copied(),
        |t| [la(t[0]), la(t[1]), la(t[2])],
        |t| (la(t[0]) + la(t[1]) + la(t[2])) / 2 + 1,
        |t| {
            let list = [[t[0] as u32, t[1] as u32, t[2] as u32]];
            evaluate_3c2e_triple_batch(&backend, &aux_shells, &list)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );
    let three_c2e_deriv = census(
        triples.iter().copied(),
        |t| [la(t[0]), la(t[1]), la(t[2])],
        |t| (la(t[0]) + la(t[1]) + la(t[2]) + 1) / 2 + 1,
        |t| {
            let list = [[t[0] as u32, t[1] as u32, t[2] as u32]];
            evaluate_3c2e_deriv_triple_batch(&backend, ThreeC2eDerivFamily::Ip1, &aux_shells, &list)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );

    let ceiling = |family| device_nroots_ceiling(&backend, family);
    json!({
        "workload": label,
        "shells": nbas,
        "auxiliary_basis": aux.name(),
        "auxiliary_shells": aux_arrays.nbas() - nbas,
        "families": [
            two_e.to_json("int2e", ceiling(RysFamily::Int2e)),
            one_e.to_json("int1e_nuc", ceiling(RysFamily::Int1e)),
            one_e_deriv.to_json("int1e_ipnuc", ceiling(RysFamily::Int1eDeriv)),
            three_c2e.to_json("int3c2e", ceiling(RysFamily::Int3c2e)),
            three_c2e_deriv.to_json("int3c2e_ip1", ceiling(RysFamily::Int3c2eDeriv)),
        ],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// The gates
// ─────────────────────────────────────────────────────────────────────────────

/// **G1.** No def2-SVP class exceeds the base device envelope, for any family.
///
/// This is the claim the plan opens with — SVP's problem is throughput, not
/// coverage — and it has to hold with the extended feature *off*, which is what
/// makes it a statement about the basis rather than about a build flag. The
/// derivative families are included because their shapes add a unit of angular
/// momentum and so cross the envelope one class earlier than the scalar ones.
#[test]
fn def2_svp_needs_no_extended_rys() {
    for (label, molecule) in [
        ("H2O", water(StandardBasis::Def2Svp)),
        ("SO2", sulfur_dioxide(StandardBasis::Def2Svp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let nbas = arrays.nbas();
        let l = |s: usize| shell_l(&arrays, s);
        let l_max = (0..nbas).map(l).max().unwrap_or(0);

        // 2e: the widest scalar shape. Its derivative adds one.
        let worst_2e = 4 * l_max / 2 + 1;
        let worst_2e_deriv = (4 * l_max + 1) / 2 + 1;
        // 1e and its gradient.
        let worst_1e = 2 * l_max / 2 + 1;
        let worst_1e_deriv = (2 * l_max).div_ceil(2) + 1;
        // The GIAO nuclear engine carries `nmax = li + lj + 5`, the largest 1e
        // headroom in the manifest — it is the family that crosses the envelope
        // first, so leaving it out would make this gate weaker than it looks.
        let worst_giao = (2 * l_max + 5) / 2 + 1;

        for (family, nroots) in [
            ("int2e", worst_2e),
            ("int2e_ip1", worst_2e_deriv),
            ("int1e_nuc", worst_1e),
            ("int1e_ipnuc", worst_1e_deriv),
            ("int1e_ignuc", worst_giao),
        ] {
            println!("def2-SVP/{label} l_max={l_max} {family}: worst nroots={nroots}");
            assert!(
                nroots <= cintx_cubecl::BASE_DEVICE_NROOTS,
                "def2-SVP/{label} {family} reaches nroots={nroots}, past the base \
                 envelope {} — def2-SVP is supposed to need no extended Rys at all",
                cintx_cubecl::BASE_DEVICE_NROOTS
            );
        }
    }
}

/// **G2.** With `extended-device-rys` on and the FMA probe passing, no class of
/// a def2 work list is refused — for any family, in either basis.
///
/// Runs one representative tuple per launch class through the real batch
/// surface, so a refusal here is the refusal a caller would get.
#[cfg(feature = "extended-device-rys")]
#[test]
fn def2_work_lists_are_fully_device_covered() {
    let backend = cpu_backend();
    assert!(
        fma_fusion_verified(&backend),
        "precondition: the CPU FMA probe passes, so the extended ceiling is live"
    );

    let mut workloads = Vec::new();
    let mut total_refused = 0_usize;
    for (label, molecule, aux) in [
        (
            "H2O/def2-SVP",
            water(StandardBasis::Def2Svp),
            StandardBasis::Def2JFit,
        ),
        (
            "H2O/def2-TZVP",
            water(StandardBasis::Def2Tzvp),
            StandardBasis::Def2JFit,
        ),
        (
            "SO2/def2-SVP",
            sulfur_dioxide(StandardBasis::Def2Svp),
            StandardBasis::Def2JFit,
        ),
        (
            "SO2/def2-TZVP",
            sulfur_dioxide(StandardBasis::Def2Tzvp),
            StandardBasis::Def2JFit,
        ),
    ] {
        let entry = coverage_for(label, &molecule, aux);
        for family in entry["families"].as_array().unwrap() {
            let refused = family["refused"].as_u64().unwrap() as usize;
            total_refused += refused;
            println!(
                "  {label:<14} {:<12} ceiling={} classes={} above_base={} accepted={} refused={} orders={:?}",
                family["family"].as_str().unwrap(),
                family["device_nroots_ceiling"],
                family["classes"],
                family["classes_above_base_ceiling"],
                family["accepted"],
                refused,
                family["nroots_present"],
            );
            for refusal in family["refusals"].as_array().unwrap() {
                eprintln!(
                    "    REFUSED {} class {}: {}",
                    family["family"].as_str().unwrap(),
                    refusal["class"],
                    refusal["detail"]
                );
            }
        }
        workloads.push(entry);
    }

    // The census must not be vacuous: TZVP is here precisely because it reaches
    // past the base ceiling, and if it stopped doing so this gate would be
    // asserting nothing.
    let above_base: u64 = workloads
        .iter()
        .flat_map(|w| w["families"].as_array().unwrap())
        .map(|f| f["classes_above_base_ceiling"].as_u64().unwrap())
        .sum();
    assert!(
        above_base > 0,
        "no def2 class exceeds the base ceiling, so this gate proves nothing about \
         the extended path"
    );
    println!("classes above the base ceiling across all workloads: {above_base}");

    let artifact = json!({
        "schema": "cintx_def2_coverage/1",
        "backend": "cpu",
        "fma_fusion_verified": true,
        "base_nroots_ceiling": cintx_cubecl::BASE_DEVICE_NROOTS,
        "extended_nroots_ceiling": cintx_cubecl::EXTENDED_DEVICE_NROOTS,
        "classes_above_base_ceiling": above_base,
        "workloads": workloads,
    });
    let written = cintx_oracle::fixtures::write_pretty_json_artifact(
        "/mnt/data/cintx_def2_coverage.json",
        "cintx_def2_coverage.json",
        &artifact,
    )
    .expect("write coverage artifact");
    println!("coverage artifact: {}", written.actual_path.display());

    assert_eq!(
        total_refused,
        0,
        "{total_refused} def2 launch classes were refused by a batch surface; \
         see the REFUSED lines above and {}",
        written.actual_path.display()
    );
}

/// The complement, and the reason G2 is a claim about a build rather than about
/// the library in general: **without** the feature, the same TZVP classes are
/// refused, explicitly and with a typed message.
///
/// A backend whose FMA probe fails lands in this state by design, and the plan
/// (D1.3) requires that to be a *reported* status rather than a silent
/// degradation. This test is what makes the report true.
#[cfg(not(feature = "extended-device-rys"))]
#[test]
fn def2_tzvp_is_refused_not_degraded_without_the_extended_path() {
    let entry = coverage_for(
        "H2O/def2-TZVP",
        &water(StandardBasis::Def2Tzvp),
        StandardBasis::Def2JFit,
    );
    let mut refused_families = Vec::new();
    for family in entry["families"].as_array().unwrap() {
        let name = family["family"].as_str().unwrap();
        let above = family["classes_above_base_ceiling"].as_u64().unwrap();
        let refused = family["refused"].as_u64().unwrap();
        println!("  {name:<12} above_base={above} refused={refused}");
        assert_eq!(
            above, refused,
            "{name}: every class above the base ceiling must be refused when the \
             extended path is not compiled in — never evaluated at a lower order"
        );
        if refused > 0 {
            refused_families.push(name.to_owned());
        }
    }
    assert!(
        !refused_families.is_empty(),
        "def2-TZVP must have classes the base ceiling cannot serve; otherwise the \
         extended path has nothing to unlock and this gate is vacuous"
    );
    println!("refused without the feature: {refused_families:?}");
}

/// The second-row fixture is the one D0.1 asks for, so its composition is
/// pinned: if the catalog ever stopped giving sulfur its `f` shell, the heavy
/// classes would quietly leave the benchmark and every TZVP timing after that
/// would be measuring a lighter workload under the same name.
#[test]
fn so2_def2_tzvp_carries_f_shells_on_sulfur() {
    let arrays = to_raw_arrays(&sulfur_dioxide(StandardBasis::Def2Tzvp)).expect("raw arrays");
    let mut on_sulfur: Vec<usize> = Vec::new();
    for shell in 0..arrays.nbas() {
        if arrays.bas[shell * BAS_SLOTS + ATOM_OF] == 0 {
            on_sulfur.push(shell_l(&arrays, shell));
        }
    }
    assert_eq!(
        on_sulfur,
        vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3],
        "def2-TZVP sulfur composition"
    );
    println!(
        "SO2/def2-TZVP: {} shells, sulfur l = {:?}",
        arrays.nbas(),
        on_sulfur
    );
}
