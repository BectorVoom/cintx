//! Phase 33, task 33-03 — `int2e` is the second family flipped onto the inline
//! extended Rys path.
//!
//! It follows `int3c2e` by workload weight: `int2e` is where a direct-SCF Fock
//! build spends its time, and its `(f f | f f)` classes are `nroots = 7`. Until
//! this flip those classes either fell back to a host primitive loop (the
//! per-tuple path) or were refused outright (the batch path).
//!
//! # Why this family is the better arm coverage
//!
//! `nroots = (li + lj + lk + ll) / 2 + 1` sums over *four* shells, so it reaches
//! the whole 6..=12 range at modest angular momenta — order 12 needs a sum of
//! 22, which `(5,5,6,6)` gives. That covers every arm of the inline dispatch
//! (f64 Jacobi and Schmidt, dd Schmidt at order 8, dd Jacobi/Laguerre at 9..12)
//! through a real family rather than only through `rys_ext_inline_parity`'s
//! direct solver sweep.
//!
//! Both entry points are gated:
//!
//! * the **batch** path (`evaluate_2e_quartet_batch`), whose ceiling check is in
//!   `evaluate_2e_batch_inner`; and
//! * the **per-tuple** path (`eval_raw` -> `launch_two_electron`), where the
//!   ceiling decides between the device kernel and the host primitive loop.
//!
//! The per-tuple gate is the more interesting of the two, because before this
//! flip that path already produced *correct* answers for `nroots >= 6` — on the
//! host. So its assertion is not "does it work now" but "does the device answer
//! the host answer", which is a tighter claim and is checked as such.

#![cfg(all(feature = "cpu", feature = "extended-device-rys", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling,
};
use cintx_cubecl::transform::c2s::C2S_LMAX;
use cintx_cubecl::{BatchShell, evaluate_2e_quartet_batch};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Absolute floor. The extended path's double-double arms round the last f64
/// bit differently from the vendor's 80-bit `long double`; the relative term is
/// what binds at large magnitudes.
const ATOL: f64 = 1e-11;

/// Relative tolerance — the dd-vs-f80 floor `rys_nroots_sweep_parity` measured
/// on the roots themselves.
const RTOL: f64 = 1e-9;

/// One class per extended Rys order.
const SYNTHETIC_CLASSES: [[i32; 4]; 7] = [
    [2, 2, 3, 3], // nroots 6
    [3, 3, 3, 3], // nroots 7  — the def2-TZVP (f f | f f) shape
    [3, 3, 4, 4], // nroots 8  — the only order whose large-x arm is dd Schmidt
    [4, 4, 4, 4], // nroots 9  — dd Jacobi below the breakpoint, dd Laguerre above
    [4, 4, 5, 5], // nroots 10
    [5, 5, 5, 5], // nroots 11
    [5, 5, 6, 6], // nroots 12 — the vendor's own quadmath-free ceiling
];

/// Four single-primitive shells on four centres at the requested angular
/// momenta. Exponents and centres differ per shell so that no accidental
/// symmetry can make a wrong root set look right.
fn synthetic_quartet(ls: [i32; 4]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.3],
        [0.0, 1.1, 0.4],
        [0.9, 0.2, 0.7],
    ];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 4];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut atm = vec![0_i32; 4 * ATM_SLOTS];
    for index in 0..4 {
        atm[index * ATM_SLOTS + CHARGE_OF] = 6;
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 4 * BAS_SLOTS];
    for (index, &l) in ls.iter().enumerate() {
        let exp_ptr = env.len() as i32;
        env.push(0.7 + 0.25 * index as f64);
        let coeff_ptr = env.len() as i32;
        env.push(1.0);
        bas[index * BAS_SLOTS + ATOM_OF] = index as i32;
        bas[index * BAS_SLOTS + ANG_OF] = l;
        bas[index * BAS_SLOTS + NPRIM_OF] = 1;
        bas[index * BAS_SLOTS + NCTR_OF] = 1;
        bas[index * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[index * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn batch_shells(ls: [i32; 4], env: &[f64], bas: &[i32], atm: &[i32]) -> Vec<BatchShell> {
    (0..4)
        .map(|index| {
            let record = &bas[index * BAS_SLOTS..(index + 1) * BAS_SLOTS];
            let exp_ptr = record[PTR_EXP] as usize;
            let coeff_ptr = record[PTR_COEFF] as usize;
            let coord_ptr = atm[index * ATM_SLOTS + PTR_COORD] as usize;
            BatchShell {
                l: ls[index] as u8,
                nprim: 1,
                nctr: 1,
                exponents: vec![env[exp_ptr]],
                coefficients: vec![env[coeff_ptr]],
                center: [env[coord_ptr], env[coord_ptr + 1], env[coord_ptr + 2]],
            }
        })
        .collect()
}

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// The precondition: `int2e` is on the flipped list, and a family that is not
/// still keeps the base ceiling in the same build.
#[test]
fn ext_rys_ceiling_is_raised_for_int2e() {
    let backend = cpu_backend();
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int2e),
        EXTENDED_DEVICE_NROOTS
    );
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int3c2eDeriv),
        BASE_DEVICE_NROOTS,
        "the 3c2e derivative set has not been flipped and must keep the base ceiling"
    );
}

/// **The gate for the batch path.** One synthetic quartet per extended Rys
/// order 6..=9, compared against vendored libcint.
#[test]
fn ext_rys_2e_batch_matches_vendor() {
    let backend = cpu_backend();
    let mut orders_covered: BTreeSet<usize> = BTreeSet::new();
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;

    for ls in SYNTHETIC_CLASSES {
        let nroots = ls.iter().sum::<i32>() as usize / 2 + 1;
        assert!(
            ls.iter().all(|&l| l <= i32::from(C2S_LMAX)),
            "class {ls:?} carries l past the c2s table ceiling {C2S_LMAX}"
        );
        let (atm, bas, env) = synthetic_quartet(ls);
        let len: usize = ls.iter().map(|&l| nsph(l)).product();

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2e_sph(&mut expected, &[0, 1, 2, 3], &atm, 4, &bas, 4, &env);

        let shells = batch_shells(ls, &env, &bas, &atm);
        let batch = evaluate_2e_quartet_batch(&backend, &shells, &[[0, 1, 2, 3]])
            .unwrap_or_else(|e| panic!("class {ls:?} (nroots={nroots}) batch failed: {e}"));
        let actual = &batch.values[batch.offsets[0]..batch.offsets[0] + len];

        let mut worst = 0.0_f64;
        let mut class_mismatches = 0_usize;
        for (e, a) in expected.iter().zip(actual) {
            compared += 1;
            let diff = (e - a).abs();
            let tol = ATOL.max(RTOL * e.abs());
            worst = worst.max(diff / tol);
            if diff > tol {
                class_mismatches += 1;
                if class_mismatches <= 5 {
                    eprintln!(
                        "  MISMATCH l={ls:?} nroots={nroots}: vendor={e:.15e} \
                         cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}"
                    );
                }
            }
        }
        mismatches += class_mismatches;
        orders_covered.insert(nroots);
        println!(
            "  l={ls:?} nroots={nroots} elements={len} worst |diff|/tol={worst:.3} \
             mismatches={class_mismatches}"
        );
    }

    assert_eq!(
        orders_covered,
        (BASE_DEVICE_NROOTS + 1..=EXTENDED_DEVICE_NROOTS).collect::<BTreeSet<_>>(),
        "the sweep must reach every extended Rys order"
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} extended-Rys 2e batch elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// **The gate for the per-tuple path.** Before this flip `eval_raw` served
/// `nroots >= 6` from a host primitive loop, so the claim here is specifically
/// that the device now answers what the host used to: the same vendor
/// comparison, through `launch_two_electron`'s device branch.
#[test]
fn ext_rys_2e_per_tuple_matches_vendor() {
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;

    for ls in SYNTHETIC_CLASSES {
        let nroots = ls.iter().sum::<i32>() as usize / 2 + 1;
        let (atm, bas, env) = synthetic_quartet(ls);
        let len: usize = ls.iter().map(|&l| nsph(l)).product();

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2e_sph(&mut expected, &[0, 1, 2, 3], &atm, 4, &bas, 4, &env);

        let mut actual = vec![0.0_f64; len];
        let status = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_sph"),
                Some(&mut actual),
                None,
                &[0, 1, 2, 3],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        };
        assert!(
            status.is_ok(),
            "per-tuple int2e class {ls:?} (nroots={nroots}) was refused: {status:?}"
        );

        let mut class_mismatches = 0_usize;
        for (e, a) in expected.iter().zip(&actual) {
            compared += 1;
            let tol = ATOL.max(RTOL * e.abs());
            if (e - a).abs() > tol {
                class_mismatches += 1;
                if class_mismatches <= 5 {
                    eprintln!(
                        "  MISMATCH l={ls:?} nroots={nroots}: vendor={e:.15e} cintx={a:.15e}"
                    );
                }
            }
        }
        mismatches += class_mismatches;
        println!("  per-tuple l={ls:?} nroots={nroots} mismatches={class_mismatches}");
    }

    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} per-tuple extended-Rys 2e elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// The batch and per-tuple paths execute the same kernel, so at these orders
/// they must agree bit for bit — a stronger statement than either vendor
/// comparison, and the one that catches a launcher plumbing the tables or the
/// order differently between the two entry points.
#[test]
fn ext_rys_2e_batch_and_per_tuple_are_bit_identical() {
    let backend = cpu_backend();
    let mut compared = 0_usize;
    for ls in SYNTHETIC_CLASSES {
        let (atm, bas, env) = synthetic_quartet(ls);
        let len: usize = ls.iter().map(|&l| nsph(l)).product();

        let shells = batch_shells(ls, &env, &bas, &atm);
        let batch = evaluate_2e_quartet_batch(&backend, &shells, &[[0, 1, 2, 3]]).expect("batch");
        let batched = &batch.values[batch.offsets[0]..batch.offsets[0] + len];

        let mut single = vec![0.0_f64; len];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_sph"),
                Some(&mut single),
                None,
                &[0, 1, 2, 3],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        }
        .expect("per-tuple");

        for (index, (b, s)) in batched.iter().zip(&single).enumerate() {
            compared += 1;
            assert_eq!(
                b.to_bits(),
                s.to_bits(),
                "class {ls:?} element {index}: batch={b:.17e} per-tuple={s:.17e}"
            );
        }
    }
    println!("batch vs per-tuple: {compared} elements bit-identical");
}
