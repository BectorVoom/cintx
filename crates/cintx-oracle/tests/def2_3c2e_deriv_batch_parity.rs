//! Task 35-D — the `int3c2e_ip1` / `int3c2e_ip2` batched path.
//!
//! Before this, every derivative triple cost its own kernel launch. On an RI-J
//! *gradient* list — `nbas^2 x naux` triples, evaluated once per geometry step —
//! that is the launch-per-tuple cost Phase 35 removed for the scalar families
//! and left in place for the derivative ones.
//!
//! Two claims, kept separate:
//!
//! 1. **The batch equals the per-triple path, bit for bit.** Not "within a
//!    tolerance": both go through the same kernel now (the per-triple entry
//!    point is a one-triple launch group), so anything but bit-identity would
//!    mean the batching itself changed a result.
//! 2. **The batch equals vendored libcint**, which is what makes claim 1 worth
//!    having — two cintx paths agreeing on a wrong answer would satisfy it.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{BatchShell, ThreeC2eDerivFamily, evaluate_3c2e_deriv_triple_batch};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Flat bound: these are the same arithmetic through two engines.
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

fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    let mut shells = Vec::with_capacity(arrays.nbas());
    for shell in 0..arrays.nbas() {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;
        let atom = record[ATOM_OF] as usize;
        let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;

        // `env` holds coefficients contraction-major; `BatchShell` wants them
        // primitive-major.
        let mut coefficients = vec![0.0_f64; nprim * nctr];
        for c in 0..nctr {
            for p in 0..nprim {
                coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
            }
        }

        shells.push(BatchShell {
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

fn shell_ao(arrays: &RawArrays, shell: usize) -> usize {
    let l = arrays.bas[shell * BAS_SLOTS + ANG_OF] as usize;
    let nctr = arrays.bas[shell * BAS_SLOTS + NCTR_OF] as usize;
    (2 * l + 1) * nctr
}

fn case(family: ThreeC2eDerivFamily) -> (RawApiId, &'static str) {
    match family {
        ThreeC2eDerivFamily::Ip1 => (RawApiId::Symbol("int3c2e_ip1_sph"), "int3c2e_ip1_sph"),
        ThreeC2eDerivFamily::Ip2 => (RawApiId::Symbol("int3c2e_ip2_sph"), "int3c2e_ip2_sph"),
    }
}

fn vendor_eval(family: ThreeC2eDerivFamily, out: &mut [f64], shls: &[i32; 3], arrays: &RawArrays) {
    match family {
        ThreeC2eDerivFamily::Ip1 => vendor_ffi::vendor_int3c2e_ip1_sph(
            out,
            shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        ),
        ThreeC2eDerivFamily::Ip2 => vendor_ffi::vendor_int3c2e_ip2_sph(
            out,
            shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        ),
    };
}

#[test]
fn def2_svp_3c2e_deriv_batch_matches_vendor_and_per_triple() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = arrays.nbas();
    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();
    assert!(!list.is_empty());

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let (api, label) = case(family);
        let batched = evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &list)
            .unwrap_or_else(|error| panic!("{label} batch failed: {error}"));

        // One dispatch per Rys order, not per `(li,lj,lk)` class and not per
        // triple. The derivative headroom shifts the order relative to the
        // scalar family — ip1 raises the bra, ip2 the auxiliary centre — so the
        // expected count is computed from the same rule the kernel uses.
        let classes: BTreeSet<[u8; 3]> = list
            .iter()
            .map(|t| {
                [
                    shells[t[0] as usize].l,
                    shells[t[1] as usize].l,
                    shells[t[2] as usize].l,
                ]
            })
            .collect();
        let dispatches: BTreeSet<usize> = classes
            .iter()
            .map(|&[li, lj, lk]| {
                let (a, b, c) = match family {
                    ThreeC2eDerivFamily::Ip1 => (li as usize + 1, lj as usize, lk as usize),
                    ThreeC2eDerivFamily::Ip2 => (li as usize, lj as usize, lk as usize + 1),
                };
                (a + b + c) / 2 + 1
            })
            .collect();

        assert_eq!(
            batched.stats.launch_classes,
            classes.len(),
            "{label} classes"
        );
        assert_eq!(
            batched.stats.kernel_launch_count,
            dispatches.len(),
            "{label}: expected one dispatch per Rys order"
        );
        assert!(
            batched.stats.kernel_launch_count < classes.len(),
            "{label}: merging must dispatch fewer times than there are classes: {} vs {}",
            batched.stats.kernel_launch_count,
            classes.len()
        );
        assert!(
            batched.stats.kernel_launch_count < list.len(),
            "{label}: batching must reduce launches below the triple count"
        );
        assert_eq!(batched.stats.readback_count, dispatches.len());
        assert_eq!(batched.stats.quartets, list.len());

        let mut vendor_mismatches = 0_usize;
        let mut single_mismatches = 0_usize;
        let mut max_vendor_diff = 0.0_f64;
        let mut first = String::new();

        let mut expected = vec![0.0_f64; 8192];
        let mut single = vec![0.0_f64; 8192];
        for (index, triple) in list.iter().enumerate() {
            let len = 3
                * shell_ao(&arrays, triple[0] as usize)
                * shell_ao(&arrays, triple[1] as usize)
                * shell_ao(&arrays, triple[2] as usize);
            if expected.len() < len {
                expected.resize(len, 0.0);
                single.resize(len, 0.0);
            }
            let shls = [triple[0] as i32, triple[1] as i32, triple[2] as i32];

            vendor_eval(family, &mut expected[..len], &shls, &arrays);

            // The per-triple compatibility path, which is a one-triple launch
            // group through the same kernel.
            let status = unsafe {
                eval_raw(
                    api,
                    Some(&mut single[..len]),
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
                "{label}: per-triple {shls:?} failed: {status:?}"
            );

            let start = batched.offsets[index];
            for element in 0..len {
                let batch_value = batched.values[start + element];

                let diff = (expected[element] - batch_value).abs();
                if diff > max_vendor_diff {
                    max_vendor_diff = diff;
                }
                if diff > TOLERANCE {
                    vendor_mismatches += 1;
                    if first.is_empty() {
                        first = format!(
                            "{label} {shls:?} elem {element}: vendor={:.17e} batch={batch_value:.17e}",
                            expected[element]
                        );
                    }
                }

                // Bit-identity against the per-triple path: same kernel, so
                // anything else means the batching moved a result.
                if single[element].to_bits() != batch_value.to_bits() {
                    single_mismatches += 1;
                    if first.is_empty() {
                        first = format!(
                            "{label} {shls:?} elem {element}: per-triple={:.17e} batch={batch_value:.17e}",
                            single[element]
                        );
                    }
                }
            }
        }

        println!(
            "{label}: triples={}  classes={}  dispatches={}  max|diff| vs vendor={max_vendor_diff:.3e}",
            list.len(),
            classes.len(),
            dispatches.len()
        );
        assert_eq!(
            single_mismatches, 0,
            "{label}: {single_mismatches} elements differ from the per-triple path. {first}"
        );
        assert_eq!(
            vendor_mismatches, 0,
            "{label}: {vendor_mismatches} elements exceed {TOLERANCE:.0e} vs vendored libcint \
             (max|diff| {max_vendor_diff:.3e}). {first}"
        );
    }
}

/// An empty list is a no-op, not a launch.
#[test]
fn empty_3c2e_deriv_batch_is_a_no_op() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let output =
            evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &[]).expect("empty batch");
        assert!(output.values.is_empty());
        assert!(output.offsets.is_empty());
        assert_eq!(output.stats.kernel_launch_count, 0);
    }
}

/// A shell index past the basis is rejected before any device work.
#[test]
fn deriv_batch_rejects_an_out_of_range_shell() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");
    let out_of_range = shells.len() as u32;

    let result = evaluate_3c2e_deriv_triple_batch(
        &backend,
        ThreeC2eDerivFamily::Ip1,
        &shells,
        &[[0, 0, out_of_range]],
    );
    assert!(result.is_err(), "an out-of-range shell must be rejected");
}

/// What the batching bought, on the list an RI-J gradient actually evaluates.
///
/// Reported against vendored libcint over the identical list, and against the
/// per-triple cintx path, so the launch saving is separated from the arithmetic.
#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_3c2e_deriv_batched_throughput() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = arrays.nbas();
    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let mut lengths = Vec::with_capacity(list.len());
    for triple in &list {
        lengths.push(
            3 * shell_ao(&arrays, triple[0] as usize)
                * shell_ao(&arrays, triple[1] as usize)
                * shell_ao(&arrays, triple[2] as usize),
        );
    }
    const REPEATS: usize = 9;

    println!(
        "\nH2O / def2-SVP  int3c2e_ip*  ({nbas} shells, {} triples)",
        list.len()
    );
    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let (api, label) = case(family);

        let mut scratch = vec![0.0_f64; 8192];
        let mut vendor_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for (index, triple) in list.iter().enumerate() {
                let len = lengths[index];
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                vendor_eval(
                    family,
                    &mut scratch[..len],
                    &[triple[0] as i32, triple[1] as i32, triple[2] as i32],
                    &arrays,
                );
            }
            vendor_secs = vendor_secs.min(start.elapsed().as_secs_f64());
        }

        let mut single_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for (index, triple) in list.iter().enumerate() {
                let len = lengths[index];
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                let _ = unsafe {
                    eval_raw(
                        api,
                        Some(&mut scratch[..len]),
                        None,
                        &[triple[0] as i32, triple[1] as i32, triple[2] as i32],
                        &arrays.atm,
                        &arrays.bas,
                        &arrays.env,
                        None,
                        None,
                    )
                };
            }
            single_secs = single_secs.min(start.elapsed().as_secs_f64());
        }

        // Warm-up pays the per-signature CubeCL specialization outside the timer.
        let mut batched =
            evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &list).expect("warm-up");
        let mut batch_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            batched = evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &list)
                .expect("batched");
            batch_secs = batch_secs.min(start.elapsed().as_secs_f64());
        }

        println!(
            "  {label:<18} libcint {vendor_secs:.5} s   cintx per-triple {single_secs:.5} s   \
cintx batched {batch_secs:.5} s ({} launches, {} classes)   speed-up {:.1}x   vs libcint {:.2}x",
            batched.stats.kernel_launch_count,
            batched.stats.launch_classes,
            single_secs / batch_secs.max(f64::MIN_POSITIVE),
            batch_secs / vendor_secs.max(f64::MIN_POSITIVE),
        );
        println!(
            "  {:<18} split: backend dispatch {:.3} ms  host cart->sph {:.3} ms",
            "",
            batched.stats.dispatch_ns as f64 / 1e6,
            batched.stats.host_transform_ns as f64 / 1e6,
        );
        if let Some(split) = cintx_cubecl::transform::profile::format_split(&batched.stats) {
            println!("  {:<18} {split}", "");
        }
    }
}
