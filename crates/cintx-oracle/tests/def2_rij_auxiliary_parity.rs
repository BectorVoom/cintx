//! def2/J and def2/JK auxiliary-basis gate (post-Phase-35 plan, Part 4).
//!
//! Two things are established here, and they are separate claims:
//!
//! 1. **Correctness.** Every auxiliary shell in the catalogue round-trips
//!    through `to_raw_arrays_with_auxiliary` and reproduces vendored libcint on
//!    `int2c2e` (the `(P|Q)` metric) and `int3c2e` (the `(mu nu | P)` tensor).
//!    Auxiliary sets reach angular momenta the AO fixtures do not — def2/J
//!    hydrogen already carries a `d` shell — so this covers launch classes the
//!    def2-SVP AO tests never produced.
//!
//! 2. **Throughput on the shape that matters.** RI-J's work list is
//!    `nbas^2 x naux`, not `nbas^3`. Benchmarking `int3c2e` over an AO-only
//!    triple list measured the kernel on a list no RI-J build ever evaluates.
//!    The ignored benchmark below builds the real one.
//!
//! Neither claim depends on the other: the parity tests run in CI, the
//! benchmark is `#[ignore]`d and run explicitly.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::raw::{ANG_OF, BAS_SLOTS, NCTR_OF};
use cintx_basis::{
    AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays, to_raw_arrays_with_auxiliary,
};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_oracle::vendor_ffi;
use std::collections::BTreeSet;

/// Bit-for-bit: these are the same arithmetic through two engines, not two
/// approximations of one number.
const TOLERANCE: f64 = 1e-12;

fn water(basis: StandardBasis) -> Molecule {
    Molecule::new(
        vec![
            AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
        ],
        basis,
    )
}

fn shell_l(arrays: &RawArrays, shell: usize) -> usize {
    arrays.bas[shell * BAS_SLOTS + ANG_OF] as usize
}

fn shell_nctr(arrays: &RawArrays, shell: usize) -> usize {
    arrays.bas[shell * BAS_SLOTS + NCTR_OF] as usize
}

/// Spherical AO count of one shell, contractions included.
fn shell_ao(arrays: &RawArrays, shell: usize) -> usize {
    (2 * shell_l(arrays, shell) + 1) * shell_nctr(arrays, shell)
}

/// The auxiliary block is appended after the orbital block, and both ranges are
/// non-empty and disjoint. Everything below indexes on that split, so it is
/// asserted before it is relied on.
#[test]
fn auxiliary_shells_are_appended_after_the_orbital_shells() {
    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let molecule = water(StandardBasis::Def2Svp);
        let orbital_only = to_raw_arrays(&molecule).expect("orbital arrays");
        let combined = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");

        assert_eq!(
            orbital_only.nbas(),
            orbital_only.n_orbital_shells,
            "an orbital-only emission has no auxiliary block"
        );
        assert!(orbital_only.auxiliary_shells().is_empty());

        assert_eq!(
            combined.n_orbital_shells,
            orbital_only.nbas(),
            "{}: the orbital block must be unchanged by appending an auxiliary one",
            aux.name()
        );
        assert_eq!(
            &combined.bas[..orbital_only.bas.len()],
            &orbital_only.bas[..],
            "{}: appending must not perturb an orbital `bas` row",
            aux.name()
        );
        assert!(
            !combined.auxiliary_shells().is_empty(),
            "{}: water must produce auxiliary shells",
            aux.name()
        );
    }
}

/// Fitting an orbital basis is a category error and must fail closed rather
/// than quietly producing a "calculation" of the wrong thing.
#[test]
fn an_orbital_basis_is_rejected_as_an_auxiliary_one() {
    let molecule = water(StandardBasis::Def2Svp);
    for orbital in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        assert!(
            to_raw_arrays_with_auxiliary(&molecule, orbital).is_err(),
            "{} must not be accepted as an auxiliary basis",
            orbital.name()
        );
    }
}

/// `(P|Q)` over every auxiliary shell pair, against vendored libcint.
///
/// This is the RI-J metric matrix; it is inverted once per SCF, so an error
/// here contaminates every fitted density rather than one integral.
#[test]
fn auxiliary_int2c2e_matches_vendor() {
    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let molecule = water(StandardBasis::Def2Svp);
        let arrays = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");
        let mut classes: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut max_diff = 0.0_f64;
        let mut compared = 0_usize;

        for p in arrays.auxiliary_shells() {
            for q in arrays.auxiliary_shells() {
                let len = shell_ao(&arrays, p) * shell_ao(&arrays, q);
                let shls = [p as i32, q as i32];

                let mut expected = vec![0.0_f64; len];
                vendor_ffi::vendor_int2c2e_sph(
                    &mut expected,
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                );

                let mut actual = vec![0.0_f64; len];
                let status = unsafe {
                    eval_raw(
                        RawApiId::INT2C2E_SPH,
                        Some(&mut actual),
                        None,
                        &shls,
                        &arrays.atm,
                        &arrays.bas,
                        &arrays.env,
                        None,
                        None,
                    )
                };
                assert!(
                    status.is_ok(),
                    "{}: int2c2e (P={p}, Q={q}) failed: {status:?}",
                    aux.name()
                );

                for (e, a) in expected.iter().zip(&actual) {
                    max_diff = max_diff.max((e - a).abs());
                }
                classes.insert((shell_l(&arrays, p), shell_l(&arrays, q)));
                compared += len;
            }
        }

        println!(
            "{}: int2c2e  pairs={}  elements={compared}  classes={}  max|diff|={max_diff:.3e}",
            aux.name(),
            arrays.auxiliary_shells().len().pow(2),
            classes.len()
        );
        assert!(
            max_diff <= TOLERANCE,
            "{}: int2c2e max|diff| {max_diff:.3e} exceeds {TOLERANCE:.0e}",
            aux.name()
        );
        assert!(compared > 0, "{}: nothing was compared", aux.name());
    }
}

/// `(mu nu | P)` over the full RI-J work list, against vendored libcint.
///
/// The list is exactly the shape an RI-J Fock build evaluates: every orbital
/// shell pair against every auxiliary shell.
#[test]
fn auxiliary_int3c2e_matches_vendor() {
    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let molecule = water(StandardBasis::Def2Svp);
        let arrays = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");
        let mut classes: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
        let mut max_diff = 0.0_f64;
        let mut compared = 0_usize;

        for mu in arrays.orbital_shells() {
            for nu in arrays.orbital_shells() {
                for p in arrays.auxiliary_shells() {
                    let len = shell_ao(&arrays, mu) * shell_ao(&arrays, nu) * shell_ao(&arrays, p);
                    let shls = [mu as i32, nu as i32, p as i32];

                    let mut expected = vec![0.0_f64; len];
                    vendor_ffi::vendor_int3c2e_sph(
                        &mut expected,
                        &shls,
                        &arrays.atm,
                        arrays.natm() as i32,
                        &arrays.bas,
                        arrays.nbas() as i32,
                        &arrays.env,
                    );

                    let mut actual = vec![0.0_f64; len];
                    let status = unsafe {
                        eval_raw(
                            RawApiId::Symbol("int3c2e_sph"),
                            Some(&mut actual),
                            None,
                            &shls,
                            &arrays.atm,
                            &arrays.bas,
                            &arrays.env,
                            None,
                            None,
                        )
                    };
                    assert!(
                        status.is_ok(),
                        "{}: int3c2e (mu={mu}, nu={nu}, P={p}) failed: {status:?}",
                        aux.name()
                    );

                    for (e, a) in expected.iter().zip(&actual) {
                        max_diff = max_diff.max((e - a).abs());
                    }
                    classes.insert((
                        shell_l(&arrays, mu),
                        shell_l(&arrays, nu),
                        shell_l(&arrays, p),
                    ));
                    compared += len;
                }
            }
        }

        println!(
            "{}: int3c2e  triples={}  elements={compared}  classes={}  max|diff|={max_diff:.3e}",
            aux.name(),
            arrays.orbital_shells().len().pow(2) * arrays.auxiliary_shells().len(),
            classes.len()
        );
        assert!(
            max_diff <= TOLERANCE,
            "{}: int3c2e max|diff| {max_diff:.3e} exceeds {TOLERANCE:.0e}",
            aux.name()
        );
        assert!(compared > 0, "{}: nothing was compared", aux.name());
    }
}

/// The auxiliary sets reach angular momenta the AO fixtures do not.
///
/// This is the reason the parity tests above are not redundant with the
/// existing def2-SVP ones: a `(s s | d)` triple with a `d` on the *auxiliary*
/// centre is a launch class the AO-only work lists never produced.
#[test]
fn auxiliary_bases_reach_beyond_the_ao_angular_envelope() {
    let molecule = water(StandardBasis::Def2Svp);
    let orbital = to_raw_arrays(&molecule).expect("orbital arrays");
    let ao_max = orbital
        .orbital_shells()
        .map(|s| shell_l(&orbital, s))
        .max()
        .expect("water has shells");

    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let arrays = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");
        let aux_max = arrays
            .auxiliary_shells()
            .map(|s| shell_l(&arrays, s))
            .max()
            .expect("auxiliary shells exist");
        println!(
            "{}: AO l_max={ao_max}  auxiliary l_max={aux_max}",
            aux.name()
        );
        assert!(
            aux_max > ao_max,
            "{}: auxiliary l_max {aux_max} should exceed the AO l_max {ao_max}",
            aux.name()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  RI-J throughput — the `nbas^2 x naux` work list, not `nbas^3`
// ─────────────────────────────────────────────────────────────────────────────

fn batch_shells(arrays: &RawArrays) -> Vec<cintx_cubecl::BatchShell> {
    use cintx_basis::raw::{ATOM_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP};
    let mut shells = Vec::with_capacity(arrays.nbas());
    for shell in 0..arrays.nbas() {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;
        let atom = record[ATOM_OF] as usize;
        let coord_ptr = arrays.atm[atom * cintx_basis::raw::ATM_SLOTS + PTR_COORD] as usize;

        // `env` holds coefficients contraction-major; `BatchShell` wants them
        // primitive-major.
        let mut coefficients = vec![0.0_f64; nprim * nctr];
        for c in 0..nctr {
            for p in 0..nprim {
                coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
            }
        }

        shells.push(cintx_cubecl::BatchShell {
            l: record[ANG_OF] as u8,
            nprim: nprim as u32,
            nctr: nctr as u32,
            exponents: arrays.env[exp_ptr..exp_ptr + nprim].to_vec(),
            coefficients,
            center: [
                arrays.env[coord_ptr],
                arrays.env[coord_ptr + 1],
                arrays.env[coord_ptr + 2],
            ],
        });
    }
    shells
}

fn rij_case(label: &str, molecule: &Molecule, aux: StandardBasis) {
    use cintx_cubecl::backend::ResolvedBackend;
    use cintx_cubecl::evaluate_3c2e_triple_batch;
    use cintx_runtime::{BackendIntent, BackendKind};

    let arrays = to_raw_arrays_with_auxiliary(molecule, aux).expect("combined arrays");
    let orbital = arrays.orbital_shells();
    let auxiliary = arrays.auxiliary_shells();

    // The RI-J list: every orbital pair against every auxiliary shell. `mu <= nu`
    // is the permutational symmetry a real build exploits, so the list is built
    // that way rather than over the full square — measuring `nbas^2` when a Fock
    // build evaluates `nbas(nbas+1)/2` would overstate the work by ~2x.
    let mut list: Vec<[u32; 3]> = Vec::new();
    for mu in orbital.clone() {
        for nu in mu..orbital.end {
            for p in auxiliary.clone() {
                list.push([mu as u32, nu as u32, p as u32]);
            }
        }
    }

    println!("\n{}", "=".repeat(96));
    println!("{label}  [RI-J  (mu nu | P)]");
    println!("{}", "=".repeat(96));
    println!(
        "  ao shells={}  aux shells={}  triples={}  (aux basis: {})",
        orbital.len(),
        auxiliary.len(),
        list.len(),
        aux.name()
    );

    // An `l_max = 4` auxiliary set against an `l_max = 3` orbital set reaches
    // `nroots = (3 + 3 + 4) / 2 + 1 = 6`, one past the device Rys ceiling. That
    // is the Phase 33 boundary, and the batch rejects the whole list rather than
    // silently returning zeros for part of it — so report the envelope and move
    // on instead of failing the benchmark.
    let envelope_probe = {
        let shells = batch_shells(&arrays);
        let backend = ResolvedBackend::from_intent(&BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend");
        evaluate_3c2e_triple_batch(&backend, &shells, &list)
    };
    if let Err(error) = envelope_probe {
        println!("  SKIPPED: outside the device Rys envelope — {error}");
        println!("  (raising the per-family ceiling is Phase 33; see plan Part 2.)");
        return;
    }

    let mut lengths = Vec::with_capacity(list.len());
    let mut total = 0_usize;
    for t in &list {
        let len = shell_ao(&arrays, t[0] as usize)
            * shell_ao(&arrays, t[1] as usize)
            * shell_ao(&arrays, t[2] as usize);
        lengths.push(len);
        total += len;
    }

    const REPEATS: usize = 5;

    // Reference: vendored libcint over the identical list.
    let mut reference = vec![0.0_f64; total];
    let mut ref_secs = f64::INFINITY;
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        let mut cursor = 0;
        for (index, t) in list.iter().enumerate() {
            let len = lengths[index];
            vendor_ffi::vendor_int3c2e_sph(
                &mut reference[cursor..cursor + len],
                &[t[0] as i32, t[1] as i32, t[2] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            cursor += len;
        }
        ref_secs = ref_secs.min(start.elapsed().as_secs_f64());
    }

    let shells = batch_shells(&arrays);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    // Warm-up pays the per-signature CubeCL specialization outside the timer.
    let _ = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("warm-up");

    let mut act_secs = f64::INFINITY;
    let mut batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("batched 3c2e");
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("batched 3c2e");
        act_secs = act_secs.min(start.elapsed().as_secs_f64());
    }

    let mut max_diff = 0.0_f64;
    let mut mismatches = 0_usize;
    let mut cursor = 0;
    for (index, &len) in lengths.iter().enumerate() {
        let start = batched.offsets[index];
        for element in 0..len {
            let diff = (reference[cursor + element] - batched.values[start + element]).abs();
            max_diff = max_diff.max(diff);
            if diff > 1e-9 {
                mismatches += 1;
            }
        }
        cursor += len;
    }

    println!(
        "  launches={}  l-classes={}  merge factor {:.2}x  widest scratch {} B/slot",
        batched.stats.kernel_launch_count,
        batched.stats.launch_classes,
        batched.stats.launch_classes as f64 / batched.stats.kernel_launch_count.max(1) as f64,
        batched.stats.max_g_slab_bytes,
    );
    println!("  max|diff| vs vendor={max_diff:.3e}  mismatched elements={mismatches}");
    println!(
        "  last run split: backend dispatch {:.3} ms  host cart->sph {:.3} ms",
        batched.stats.dispatch_ns as f64 / 1e6,
        batched.stats.host_transform_ns as f64 / 1e6,
    );
    if let Some(split) = cintx_cubecl::transform::profile::format_split(&batched.stats) {
        println!("  {split}");
    }
    println!(
        "\n  {:<34} {:>12} {:>16} {:>16}",
        "engine", "wall (s)", "triples/s", "us/triple"
    );
    for (name, secs) in [
        ("libcint 6.1.3 (C, 1 thread)", ref_secs),
        ("cintx CubeCL batched (cpu)", act_secs),
    ] {
        println!(
            "  {:<34} {:>12.5} {:>16.1} {:>16.3}",
            name,
            secs,
            list.len() as f64 / secs.max(f64::MIN_POSITIVE),
            secs * 1e6 / list.len().max(1) as f64
        );
    }
    if mismatches == 0 {
        let (ratio, verdict) = if act_secs > ref_secs {
            (act_secs / ref_secs, "SLOWER")
        } else {
            (ref_secs / act_secs, "FASTER")
        };
        println!(
            "\n  VERDICT: batched cintx is {ratio:.2}x {verdict} than libcint on this workload."
        );
    } else {
        println!(
            "\n  VERDICT: NOT COMPARABLE — {mismatches} elements mismatched. \
             Speed is not reported for an incorrect run."
        );
    }
}

/// The RI-J benchmark the plan's Part 4 exists to make possible.
#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_rij_throughput() {
    rij_case(
        "H2O / def2-SVP + def2/J",
        &water(StandardBasis::Def2Svp),
        StandardBasis::Def2JFit,
    );
    rij_case(
        "H2O / def2-SVP + def2/JK",
        &water(StandardBasis::Def2Svp),
        StandardBasis::Def2JkFit,
    );
    rij_case(
        "H2O / def2-TZVP + def2/J",
        &water(StandardBasis::Def2Tzvp),
        StandardBasis::Def2JFit,
    );
}
