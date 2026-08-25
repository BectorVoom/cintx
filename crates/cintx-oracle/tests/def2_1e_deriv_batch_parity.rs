//! Task 35-D — the `int1e_ipovlp` / `int1e_ipkin` / `int1e_ipnuc` batched path.
//!
//! These are the nuclear-gradient 1e integrals: a geometry optimization
//! evaluates the whole `nbas^2` list once per step. Before this each pair cost
//! its own kernel launch.
//!
//! The two non-Rys operators have **nothing left to specialize on** once the
//! shape scalars are per-pair — `op_kind` is fixed by the caller's operator — so
//! a whole work list collapses to one dispatch. `int1e_ipnuc` is a Rys
//! quadrature and keeps one per distinct order.
//!
//! Same two separate claims as the 3c2e derivative gate: bit-identity against
//! the per-pair path (same kernel, so anything else means batching moved a
//! result), and agreement with vendored libcint (which is what makes the first
//! claim worth having).

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};
use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{BatchAtom, BatchShell, OneEDerivOperator, evaluate_1e_deriv_pair_batch};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

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

fn batch_atoms(arrays: &RawArrays) -> Vec<BatchAtom> {
    (0..arrays.natm())
        .map(|atom| {
            let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;
            BatchAtom {
                charge: f64::from(arrays.atm[atom * ATM_SLOTS + CHARGE_OF]),
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect()
}

fn shell_ao(arrays: &RawArrays, shell: usize) -> usize {
    let l = arrays.bas[shell * BAS_SLOTS + ANG_OF] as usize;
    let nctr = arrays.bas[shell * BAS_SLOTS + NCTR_OF] as usize;
    (2 * l + 1) * nctr
}

fn vendor_eval(operator: OneEDerivOperator, out: &mut [f64], shls: &[i32; 2], arrays: &RawArrays) {
    let (atm, natm, bas, nbas, env) = (
        &arrays.atm,
        arrays.natm() as i32,
        &arrays.bas,
        arrays.nbas() as i32,
        &arrays.env,
    );
    match operator {
        OneEDerivOperator::IpOvlp => {
            vendor_ffi::vendor_int1e_ipovlp_sph(out, shls, atm, natm, bas, nbas, env)
        }
        OneEDerivOperator::IpKin => {
            vendor_ffi::vendor_int1e_ipkin_sph(out, shls, atm, natm, bas, nbas, env)
        }
        OneEDerivOperator::IpNuc => {
            vendor_ffi::vendor_int1e_ipnuc_sph(out, shls, atm, natm, bas, nbas, env)
        }
    };
}

#[test]
fn def2_svp_1e_deriv_batch_matches_vendor_and_per_pair() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let atoms = batch_atoms(&arrays);
    let nbas = arrays.nbas();
    let list: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    for operator in [
        OneEDerivOperator::IpOvlp,
        OneEDerivOperator::IpKin,
        OneEDerivOperator::IpNuc,
    ] {
        let label = operator.symbol();
        let api = RawApiId::Symbol(label);
        let batched = evaluate_1e_deriv_pair_batch(&backend, operator, &shells, &atoms, &list)
            .unwrap_or_else(|error| panic!("{label} batch failed: {error}"));

        let classes: BTreeSet<[u8; 2]> = list
            .iter()
            .map(|p| [shells[p[0] as usize].l, shells[p[1] as usize].l])
            .collect();
        let dispatches: BTreeSet<usize> = match operator {
            OneEDerivOperator::IpNuc => classes
                .iter()
                .map(|&[li, lj]| (li as usize + lj as usize).div_ceil(2) + 1)
                .collect(),
            // Not Rys quadratures: nothing left to specialize on.
            _ => std::iter::once(1).collect(),
        };

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
            batched.stats.kernel_launch_count < list.len(),
            "{label}: batching must reduce launches below the pair count"
        );
        assert_eq!(batched.stats.readback_count, dispatches.len());
        assert_eq!(batched.stats.quartets, list.len());

        let mut vendor_mismatches = 0_usize;
        let mut single_mismatches = 0_usize;
        let mut max_vendor_diff = 0.0_f64;
        let mut first = String::new();

        let mut expected = vec![0.0_f64; 2048];
        let mut single = vec![0.0_f64; 2048];
        for (index, pair) in list.iter().enumerate() {
            let len = 3 * shell_ao(&arrays, pair[0] as usize) * shell_ao(&arrays, pair[1] as usize);
            if expected.len() < len {
                expected.resize(len, 0.0);
                single.resize(len, 0.0);
            }
            let shls = [pair[0] as i32, pair[1] as i32];

            vendor_eval(operator, &mut expected[..len], &shls, &arrays);

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
                "{label}: per-pair {shls:?} failed: {status:?}"
            );

            let start = batched.offsets[index];
            for element in 0..len {
                let value = batched.values[start + element];

                let diff = (expected[element] - value).abs();
                if diff > max_vendor_diff {
                    max_vendor_diff = diff;
                }
                if diff > TOLERANCE {
                    vendor_mismatches += 1;
                    if first.is_empty() {
                        first = format!(
                            "{label} {shls:?} elem {element}: vendor={:.17e} batch={value:.17e}",
                            expected[element]
                        );
                    }
                }

                if single[element].to_bits() != value.to_bits() {
                    single_mismatches += 1;
                    if first.is_empty() {
                        first = format!(
                            "{label} {shls:?} elem {element}: per-pair={:.17e} batch={value:.17e}",
                            single[element]
                        );
                    }
                }
            }
        }

        println!(
            "{label}: pairs={}  classes={}  dispatches={}  max|diff| vs vendor={max_vendor_diff:.3e}",
            list.len(),
            classes.len(),
            dispatches.len()
        );
        assert_eq!(
            single_mismatches, 0,
            "{label}: {single_mismatches} elements differ from the per-pair path. {first}"
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
fn empty_1e_deriv_batch_is_a_no_op() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let atoms = batch_atoms(&arrays);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    for operator in [
        OneEDerivOperator::IpOvlp,
        OneEDerivOperator::IpKin,
        OneEDerivOperator::IpNuc,
    ] {
        let output = evaluate_1e_deriv_pair_batch(&backend, operator, &shells, &atoms, &[])
            .expect("empty batch");
        assert!(output.values.is_empty());
        assert_eq!(output.stats.kernel_launch_count, 0);
    }
}

/// What the batching bought on the 1e gradient list.
#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_1e_deriv_batched_throughput() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let atoms = batch_atoms(&arrays);
    let nbas = arrays.nbas();
    let list: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let lengths: Vec<usize> = list
        .iter()
        .map(|p| 3 * shell_ao(&arrays, p[0] as usize) * shell_ao(&arrays, p[1] as usize))
        .collect();
    const REPEATS: usize = 9;

    println!(
        "\nH2O / def2-SVP  int1e_ip*  ({nbas} shells, {} pairs)",
        list.len()
    );
    for operator in [
        OneEDerivOperator::IpOvlp,
        OneEDerivOperator::IpKin,
        OneEDerivOperator::IpNuc,
    ] {
        let label = operator.symbol();
        let api = RawApiId::Symbol(label);
        let mut scratch = vec![0.0_f64; 2048];

        let mut vendor_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for (index, pair) in list.iter().enumerate() {
                let len = lengths[index];
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                vendor_eval(
                    operator,
                    &mut scratch[..len],
                    &[pair[0] as i32, pair[1] as i32],
                    &arrays,
                );
            }
            vendor_secs = vendor_secs.min(start.elapsed().as_secs_f64());
        }

        let mut single_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for (index, pair) in list.iter().enumerate() {
                let len = lengths[index];
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                let _ = unsafe {
                    eval_raw(
                        api,
                        Some(&mut scratch[..len]),
                        None,
                        &[pair[0] as i32, pair[1] as i32],
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

        // Warm-up pays the CubeCL specialization outside the timer.
        let mut batched = evaluate_1e_deriv_pair_batch(&backend, operator, &shells, &atoms, &list)
            .expect("warm-up");
        let mut batch_secs = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            batched = evaluate_1e_deriv_pair_batch(&backend, operator, &shells, &atoms, &list)
                .expect("batched");
            batch_secs = batch_secs.min(start.elapsed().as_secs_f64());
        }

        println!(
            "  {label:<18} libcint {vendor_secs:.5} s   cintx per-pair {single_secs:.5} s   \
cintx batched {batch_secs:.5} s ({} launches, {} classes)   speed-up {:.1}x   vs libcint {:.2}x",
            batched.stats.kernel_launch_count,
            batched.stats.launch_classes,
            single_secs / batch_secs.max(f64::MIN_POSITIVE),
            batch_secs / vendor_secs.max(f64::MIN_POSITIVE),
        );
    }
}
