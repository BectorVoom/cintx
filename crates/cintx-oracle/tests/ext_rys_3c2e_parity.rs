//! Phase 33, task 33-03 — `int3c2e` is the first family flipped onto the
//! inline extended Rys path.
//!
//! It goes first because it is the one with a workload waiting on it: RI-J over
//! H2O/def2-TZVP with the def2/J auxiliary set reaches
//! `nroots = (3 + 3 + 4) / 2 + 1 = 6` on its `(f f | g)` triples, and
//! `def2_rij_auxiliary_parity`'s benchmark reported that as
//! `SKIPPED: outside the device Rys envelope` rather than measuring it.
//!
//! # What the gate is
//!
//! Vendored libcint 6.1.3 over exactly the triples whose Rys order exceeds the
//! polynomial-fit ceiling — nothing else. The `nroots <= 5` classes are already
//! covered by `def2_rij_auxiliary_parity` and `center_3c2e_parity`; repeating
//! them here would dilute a failure signal that should point at one thing.
//!
//! Both entry points are gated, because they reach the kernel by different
//! routes and each had its own ceiling to raise:
//!
//! * the **batch** path (`evaluate_3c2e_triple_batch`), whose ceiling check
//!   lives in `evaluate_3c2e_triple_batch_with`; and
//! * the **per-tuple** path (`eval_raw` -> `launch_center_3c2e`), whose check
//!   lives in `launch_center_3c2e_typed`.
//!
//! # Why this file is feature-gated rather than always compiled
//!
//! `extended-device-rys` is off by default and, even when on, effective only
//! where `device_rys_ceiling::fma_fusion_verified` passes for the backend. A
//! test that ran without the feature would be asserting the *old* behaviour, so
//! it is the feature that selects which suite applies, not a runtime branch.

#![cfg(all(feature = "cpu", feature = "extended-device-rys", has_vendor_libcint))]

use cintx_basis::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays_with_auxiliary};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling,
};
use cintx_cubecl::evaluate_3c2e_triple_batch;
use cintx_cubecl::transform::c2s::C2S_LMAX;
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Absolute floor. The extended path's double-double arms round the last f64
/// bit differently from the vendor's 80-bit `long double`, so an integral that
/// is O(1) can sit a few ulp away; the relative term below is what actually
/// binds at large magnitudes.
const ATOL: f64 = 1e-11;

/// Relative tolerance, the dd-vs-f80 floor `rys_nroots_sweep_parity` measured
/// on the roots themselves.
const RTOL: f64 = 1e-9;

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

fn shell_ao(arrays: &RawArrays, shell: usize) -> usize {
    let l = shell_l(arrays, shell);
    let nctr = arrays.bas[shell * BAS_SLOTS + NCTR_OF] as usize;
    (2 * l + 1) * nctr
}

fn batch_shells(arrays: &RawArrays) -> Vec<cintx_cubecl::BatchShell> {
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

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// Every `(mu, nu, P)` triple of the def2-TZVP + `aux` RI-J list whose Rys
/// order is past the polynomial-fit ceiling.
fn high_order_triples(arrays: &RawArrays) -> Vec<[u32; 3]> {
    let mut list = Vec::new();
    for mu in arrays.orbital_shells() {
        for nu in mu..arrays.orbital_shells().end {
            for p in arrays.auxiliary_shells() {
                let nroots =
                    (shell_l(arrays, mu) + shell_l(arrays, nu) + shell_l(arrays, p)) / 2 + 1;
                if nroots > BASE_DEVICE_NROOTS {
                    list.push([mu as u32, nu as u32, p as u32]);
                }
            }
        }
    }
    list
}

/// The precondition: with the feature compiled in, the CPU probe passing and
/// `int3c2e` on the flipped list, the ceiling really is raised for this family.
/// Without this the parity tests below would pass trivially by finding nothing
/// to compare.
///
/// The second half is the other side of the same claim: the 3c2e *derivative*
/// set has not been flipped, and keeps the base ceiling in the very same build.
/// Raising the ceiling for everyone at once is what made the then-unflipped
/// `int2e` batch accept an `(f f | f f)` class and evaluate it at order 5.
#[test]
fn ext_rys_ceiling_is_raised_for_int3c2e_and_not_its_derivatives() {
    let backend = cpu_backend();
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int3c2e),
        EXTENDED_DEVICE_NROOTS
    );
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int3c2eDeriv),
        BASE_DEVICE_NROOTS,
        "the 3c2e derivative set has not been flipped and must keep the base ceiling"
    );
}

/// **The gate for the batch path.** Every high-Rys-order `(mu nu | P)` triple
/// of H2O/def2-TZVP against def2/J and def2/JK reproduces vendored libcint.
#[test]
fn ext_rys_3c2e_batch_matches_vendor() {
    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let molecule = water(StandardBasis::Def2Tzvp);
        let arrays = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");
        let list = high_order_triples(&arrays);
        assert!(
            !list.is_empty(),
            "{}: def2-TZVP against this auxiliary set produced no class past \
             nroots={BASE_DEVICE_NROOTS}, so this gate would be vacuous",
            aux.name()
        );

        let shells = batch_shells(&arrays);
        let batch = evaluate_3c2e_triple_batch(&cpu_backend(), &shells, &list)
            .unwrap_or_else(|e| panic!("{}: high-order 3c2e batch failed: {e}", aux.name()));

        let mut classes: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
        let mut orders: BTreeSet<usize> = BTreeSet::new();
        let mut worst = 0.0_f64;
        let mut mismatches = 0_usize;
        let mut compared = 0_usize;

        for (index, t) in list.iter().enumerate() {
            let len = shell_ao(&arrays, t[0] as usize)
                * shell_ao(&arrays, t[1] as usize)
                * shell_ao(&arrays, t[2] as usize);
            let start = batch.offsets[index];
            let actual = &batch.values[start..start + len];

            let mut expected = vec![0.0_f64; len];
            vendor_ffi::vendor_int3c2e_sph(
                &mut expected,
                &[t[0] as i32, t[1] as i32, t[2] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );

            let (li, lj, lk) = (
                shell_l(&arrays, t[0] as usize),
                shell_l(&arrays, t[1] as usize),
                shell_l(&arrays, t[2] as usize),
            );
            classes.insert((li, lj, lk));
            orders.insert((li + lj + lk) / 2 + 1);

            for (e, a) in expected.iter().zip(actual) {
                compared += 1;
                let diff = (e - a).abs();
                let tol = ATOL.max(RTOL * e.abs());
                worst = worst.max(diff / tol);
                if diff > tol {
                    mismatches += 1;
                    if mismatches <= 10 {
                        eprintln!(
                            "  MISMATCH ({},{},{}) l=({li},{lj},{lk}): vendor={e:.15e} \
                             cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}",
                            t[0], t[1], t[2]
                        );
                    }
                }
            }
        }

        println!(
            "{}: extended-Rys 3c2e  triples={}  elements={compared}  classes={}  \
             nroots={orders:?}  worst |diff|/tol={worst:.3}",
            aux.name(),
            list.len(),
            classes.len()
        );
        assert_eq!(
            mismatches,
            0,
            "{}: {mismatches} of {compared} extended-Rys 3c2e elements exceeded \
             max(atol={ATOL:e}, rtol={RTOL:e})",
            aux.name()
        );
    }
}

/// **The gate for the per-tuple path.** `eval_raw` reaches the same kernel
/// through `launch_center_3c2e`, whose own ceiling check is a separate line of
/// code from the batch path's — so it gets its own gate rather than being
/// assumed to follow.
#[test]
fn ext_rys_3c2e_per_tuple_matches_vendor() {
    let molecule = water(StandardBasis::Def2Tzvp);
    let arrays =
        to_raw_arrays_with_auxiliary(&molecule, StandardBasis::Def2JFit).expect("combined arrays");
    let list = high_order_triples(&arrays);
    assert!(!list.is_empty(), "no high-order triple to compare");

    let mut mismatches = 0_usize;
    let mut compared = 0_usize;
    for t in &list {
        let len = shell_ao(&arrays, t[0] as usize)
            * shell_ao(&arrays, t[1] as usize)
            * shell_ao(&arrays, t[2] as usize);
        let shls = [t[0] as i32, t[1] as i32, t[2] as i32];

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
            "per-tuple int3c2e ({}, {}, {}) failed: {status:?}",
            t[0],
            t[1],
            t[2]
        );

        for (e, a) in expected.iter().zip(&actual) {
            compared += 1;
            let tol = ATOL.max(RTOL * e.abs());
            if (e - a).abs() > tol {
                mismatches += 1;
                if mismatches <= 10 {
                    eprintln!(
                        "  MISMATCH ({},{},{}): vendor={e:.15e} cintx={a:.15e}",
                        t[0], t[1], t[2]
                    );
                }
            }
        }
    }

    println!(
        "per-tuple extended-Rys 3c2e: triples={} elements={compared}",
        list.len()
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} per-tuple extended-Rys 3c2e elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Synthetic class sweep — the arms the real bases cannot reach
// ─────────────────────────────────────────────────────────────────────────────

/// The RI-J lists above top out at `nroots = 6`: an `l_max = 4` auxiliary set
/// against an `l_max = 3` orbital set cannot produce more. This sweep reaches
/// every remaining arm of the inline dispatch.
///
/// # Why this list is what it is
///
/// `nroots = (li + lj + lk) / 2 + 1`, so order 12 needs an angular-momentum sum
/// of 22 — `l >= 7` on two centres. That was unreachable while the
/// Cartesian-to-spherical tables stopped at `l = 4` and returned 0.0 above it;
/// with the tables generated from libcint's own `g_trans_cart2sph[]` for
/// `l = 0..=15`, spherical classes carry the whole range and each arm of the
/// dispatch — f64 Jacobi, f64 Schmidt, dd Schmidt at order 8, dd
/// Jacobi/Laguerre at 9..12 — is reachable through a real family rather than
/// only through `rys_ext_inline_parity`'s direct solver sweep.
const SYNTHETIC_CLASSES: [[i32; 3]; 7] = [
    [3, 3, 4], // nroots 6  — f f | g, the def2-TZVP + def2/J shape
    [4, 4, 4], // nroots 7
    [4, 5, 5], // nroots 8  — the only order whose large-x arm is dd Schmidt
    [5, 5, 6], // nroots 9
    [6, 6, 6], // nroots 10
    [6, 7, 7], // nroots 11
    [7, 7, 8], // nroots 12 — the vendor's own quadmath-free ceiling
];

/// Three single-primitive shells on three centres, at the requested angular
/// momenta. Exponents differ per shell so no accidental symmetry can make a
/// wrong root set look right.
fn synthetic_triple(ls: [i32; 3]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::{CHARGE_OF, NUC_MOD_OF, POINT_NUC, PTR_ENV_START, PTR_ZETA};

    let coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.9, 0.4]];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 3];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for index in 0..3 {
        atm[index * ATM_SLOTS + CHARGE_OF] = 6;
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    for (index, &l) in ls.iter().enumerate() {
        let exp_ptr = env.len() as i32;
        env.push(0.8 + 0.3 * index as f64);
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

/// Every extended Rys order, compared against vendored libcint on a synthetic
/// class per order.
#[test]
fn ext_rys_3c2e_reaches_every_arm() {
    let mut orders_covered: BTreeSet<usize> = BTreeSet::new();
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;
    let mut nonzero = 0_usize;

    for ls in SYNTHETIC_CLASSES {
        let nroots = (ls[0] + ls[1] + ls[2]) as usize / 2 + 1;
        assert!(
            nroots <= EXTENDED_DEVICE_NROOTS,
            "synthetic class {ls:?} asks for nroots={nroots}, past the extended ceiling"
        );
        assert!(
            ls.iter().all(|&l| l <= i32::from(C2S_LMAX)),
            "synthetic class {ls:?} carries l past the c2s table ceiling {C2S_LMAX}"
        );
        let (atm, bas, env) = synthetic_triple(ls);
        let len: usize = ls.iter().map(|&l| (2 * l + 1) as usize).product();

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int3c2e_sph(&mut expected, &[0, 1, 2], &atm, 3, &bas, 3, &env);

        let mut actual = vec![0.0_f64; len];
        let status = unsafe {
            eval_raw(
                RawApiId::Symbol("int3c2e_sph"),
                Some(&mut actual),
                None,
                &[0, 1, 2],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        };
        assert!(
            status.is_ok(),
            "synthetic class {ls:?} (nroots={nroots}) was refused: {status:?}"
        );

        let mut worst = 0.0_f64;
        let mut class_mismatches = 0_usize;
        for (e, a) in expected.iter().zip(&actual) {
            compared += 1;
            if *a != 0.0 && ls.iter().any(|&l| l >= 5) {
                nonzero += 1;
            }
            let diff = (e - a).abs();
            let tol = ATOL.max(RTOL * e.abs());
            worst = worst.max(diff / tol);
            if diff > tol {
                class_mismatches += 1;
                if class_mismatches <= 5 {
                    eprintln!(
                        "  MISMATCH l={ls:?} nroots={nroots}: vendor={e:.15e}                          cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}"
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
    // Agreement alone would be satisfied by both sides returning zero, which is
    // exactly what the l >= 5 c2s gap used to produce.
    assert!(
        nonzero > 0,
        "no non-zero output at l >= 5; the c2s transform is zeroing again"
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} synthetic extended-Rys 3c2e elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// Above the extended ceiling the family stays fail-closed. `nroots = 13` is
/// where the vendor itself would need quadmath, which this build does not
/// compile, so there is no reference to be compatible with and the right answer
/// is a typed refusal — not a clamp back to a lower order, which is what the
/// launcher's `_` match arm would silently do if a class ever reached it.
#[test]
fn ext_rys_3c2e_still_refuses_past_the_extended_ceiling() {
    let (atm, bas, env) = synthetic_triple([8, 8, 8]);
    let len: usize = 17 * 17 * 17;
    let mut actual = vec![0.0_f64; len];
    let status = unsafe {
        eval_raw(
            RawApiId::Symbol("int3c2e_sph"),
            Some(&mut actual),
            None,
            &[0, 1, 2],
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    };
    assert!(
        status.is_err(),
        "nroots=13 is past the extended ceiling {EXTENDED_DEVICE_NROOTS} and must be refused"
    );
}
