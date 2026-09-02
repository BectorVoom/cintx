//! `def2_speed_precision_plan.md` D1.1 — the `int3c2e` **derivative** set is
//! flipped onto the inline extended Rys path.
//!
//! # Why this family, and why now
//!
//! `int3c2e` itself was flipped in task 33-03; its derivative set was
//! deliberately left behind and used, in three sibling gates, as the live proof
//! that the ceiling really is per family. That left an RI-J *gradient* list
//! refusing exactly the triples an RI-J energy list evaluates: the derivative
//! shape raises the bra (`ip1`) or the real auxiliary (`ip2`) by one, so
//! `nroots = (li + 1 + lj + lk) / 2 + 1` crosses 5 one class *earlier* than the
//! scalar family does. H2O/def2-TZVP against def2/J reaches order 6 on triples
//! whose scalar counterparts sit at 5 and have been on the device all along.
//!
//! # What the gate is
//!
//! Vendored libcint 6.1.3 over exactly the triples whose Rys order exceeds the
//! polynomial-fit ceiling — nothing else. The `nroots <= 5` classes are covered
//! by `def2_3c2e_deriv_batch_parity`; repeating them here would dilute a failure
//! signal that should point at one thing.
//!
//! Both entry points are gated, because they reach the kernel by different
//! routes and each had its own ceiling to raise:
//!
//! * the **batch** path (`evaluate_3c2e_deriv_triple_batch`), whose check lives
//!   in `evaluate_3c2e_deriv_batch_inner`; and
//! * the **per-tuple** path (`eval_raw` -> `launch_center_3c2e_ip1` /
//!   `launch_center_3c2e_ip2`), which until this change hardcoded a `5`.
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
use cintx_cubecl::transform::c2s::C2S_LMAX;
use cintx_cubecl::{BatchShell, ThreeC2eDerivFamily, evaluate_3c2e_deriv_triple_batch};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Absolute floor. The extended path's double-double arms round the last f64
/// bit differently from the vendor's 80-bit `long double`, so an integral that
/// is O(1) can sit a few ulp away; the relative term below is what actually
/// binds at large magnitudes. Same pair the scalar `ext_rys_3c2e_parity` uses —
/// the derivative shares the solver, so it inherits the solver's floor.
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

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// The Rys order the derivative shape asks for.
///
/// `ip1` raises the bra `i`, `ip2` raises the real auxiliary `k` (which lives in
/// the 2e `ll` slot). Either way one unit of angular momentum enters the sum, so
/// this is `build_2e_shape(li + 1, lj, 0, lk)`'s `rys_order` written out.
fn deriv_nroots(li: usize, lj: usize, lk: usize) -> usize {
    (li + lj + lk + 1) / 2 + 1
}

fn api_for(family: ThreeC2eDerivFamily) -> (RawApiId, &'static str) {
    match family {
        ThreeC2eDerivFamily::Ip1 => (RawApiId::Symbol("int3c2e_ip1_sph"), "int3c2e_ip1_sph"),
        ThreeC2eDerivFamily::Ip2 => (RawApiId::Symbol("int3c2e_ip2_sph"), "int3c2e_ip2_sph"),
    }
}

fn vendor_eval(
    family: ThreeC2eDerivFamily,
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) {
    match family {
        ThreeC2eDerivFamily::Ip1 => {
            vendor_ffi::vendor_int3c2e_ip1_sph(out, shls, atm, natm, bas, nbas, env)
        }
        ThreeC2eDerivFamily::Ip2 => {
            vendor_ffi::vendor_int3c2e_ip2_sph(out, shls, atm, natm, bas, nbas, env)
        }
    };
}

/// Every `(mu, nu, P)` triple of the def2-TZVP + `aux` RI-J gradient list whose
/// derivative Rys order is past the polynomial-fit ceiling.
fn high_order_triples(arrays: &RawArrays) -> Vec<[u32; 3]> {
    let mut list = Vec::new();
    for mu in arrays.orbital_shells() {
        for nu in mu..arrays.orbital_shells().end {
            for p in arrays.auxiliary_shells() {
                let nroots =
                    deriv_nroots(shell_l(arrays, mu), shell_l(arrays, nu), shell_l(arrays, p));
                if nroots > BASE_DEVICE_NROOTS {
                    list.push([mu as u32, nu as u32, p as u32]);
                }
            }
        }
    }
    list
}

/// The precondition: with the feature compiled in, the CPU probe passing and
/// the 3c2e derivative set now on the flipped list, the ceiling really is raised
/// for this family. Without this the parity tests below would pass trivially by
/// finding nothing to compare.
///
/// The second half is the other side of the same claim, and the reason this is
/// still a per-family question rather than a global switch: the **one-electron**
/// derivative set has not been flipped, and keeps the base ceiling in the very
/// same build.
#[test]
fn ext_rys_ceiling_is_raised_for_the_3c2e_derivatives() {
    let backend = cpu_backend();
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int3c2eDeriv),
        EXTENDED_DEVICE_NROOTS
    );
    // Every declared family is now flipped (def2 plan D1), so this gate can no
    // longer point at a sibling that is still on the base ceiling. What replaces
    // that check is `device_rys_ceiling::tests::a_familys_ceiling_follows_its_own_flip`,
    // which asserts the ceiling of every family against its own
    // `runs_extended_rys` flag rather than against a neighbour's.
    assert!(
        RysFamily::Int1eDeriv.runs_extended_rys(),
        "if the 1e derivative set is ever unflipped, this gate's sibling check \
         has to come back with it"
    );
}

/// **The gate for the batch path.** Every high-Rys-order `(mu nu | P)` triple of
/// H2O/def2-TZVP against def2/J and def2/JK reproduces vendored libcint, for
/// both derivative families.
#[test]
fn ext_rys_3c2e_deriv_batch_matches_vendor() {
    for aux in [StandardBasis::Def2JFit, StandardBasis::Def2JkFit] {
        let molecule = water(StandardBasis::Def2Tzvp);
        let arrays = to_raw_arrays_with_auxiliary(&molecule, aux).expect("combined arrays");
        let list = high_order_triples(&arrays);
        assert!(
            !list.is_empty(),
            "{}: def2-TZVP against this auxiliary set produced no derivative class past \
             nroots={BASE_DEVICE_NROOTS}, so this gate would be vacuous",
            aux.name()
        );
        let shells = batch_shells(&arrays);

        for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
            let (_, label) = api_for(family);
            let batch = evaluate_3c2e_deriv_triple_batch(&cpu_backend(), family, &shells, &list)
                .unwrap_or_else(|e| panic!("{}: high-order {label} batch failed: {e}", aux.name()));

            let mut classes: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
            let mut orders: BTreeSet<usize> = BTreeSet::new();
            let mut worst = 0.0_f64;
            let mut mismatches = 0_usize;
            let mut compared = 0_usize;

            for (index, t) in list.iter().enumerate() {
                // Three Cartesian components per AO block.
                let len = 3
                    * shell_ao(&arrays, t[0] as usize)
                    * shell_ao(&arrays, t[1] as usize)
                    * shell_ao(&arrays, t[2] as usize);
                let start = batch.offsets[index];
                let actual = &batch.values[start..start + len];

                let mut expected = vec![0.0_f64; len];
                vendor_eval(
                    family,
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
                orders.insert(deriv_nroots(li, lj, lk));

                for (e, a) in expected.iter().zip(actual) {
                    compared += 1;
                    let diff = (e - a).abs();
                    let tol = ATOL.max(RTOL * e.abs());
                    worst = worst.max(diff / tol);
                    if diff > tol {
                        mismatches += 1;
                        if mismatches <= 10 {
                            eprintln!(
                                "  MISMATCH {label} ({},{},{}) l=({li},{lj},{lk}): \
                                 vendor={e:.15e} cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}",
                                t[0], t[1], t[2]
                            );
                        }
                    }
                }
            }

            println!(
                "{}: extended-Rys {label}  triples={}  elements={compared}  classes={}  \
                 nroots={orders:?}  worst |diff|/tol={worst:.3}",
                aux.name(),
                list.len(),
                classes.len()
            );
            assert_eq!(
                mismatches,
                0,
                "{}: {mismatches} of {compared} extended-Rys {label} elements exceeded \
                 max(atol={ATOL:e}, rtol={RTOL:e})",
                aux.name()
            );
        }
    }
}

/// **The gate for the per-tuple path.** `eval_raw` reaches the same kernel
/// through `launch_center_3c2e_ip1` / `_ip2`, whose ceiling check is a separate
/// line of code from the batch path's — and was a hardcoded `5` until this
/// change — so it gets its own gate rather than being assumed to follow.
#[test]
fn ext_rys_3c2e_deriv_per_tuple_matches_vendor() {
    let molecule = water(StandardBasis::Def2Tzvp);
    let arrays =
        to_raw_arrays_with_auxiliary(&molecule, StandardBasis::Def2JFit).expect("combined arrays");
    let list = high_order_triples(&arrays);
    assert!(!list.is_empty(), "no high-order triple to compare");

    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let (api, label) = api_for(family);
        let mut mismatches = 0_usize;
        let mut compared = 0_usize;

        for t in &list {
            let len = 3
                * shell_ao(&arrays, t[0] as usize)
                * shell_ao(&arrays, t[1] as usize)
                * shell_ao(&arrays, t[2] as usize);
            let shls = [t[0] as i32, t[1] as i32, t[2] as i32];

            let mut expected = vec![0.0_f64; len];
            vendor_eval(
                family,
                &mut expected,
                &shls,
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );

            let mut actual = vec![0.0_f64; len];
            // SAFETY: `actual` is sized from the vendor's own AO counts times
            // the three derivative components.
            let status = unsafe {
                eval_raw(
                    api,
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
                "per-tuple {label} ({}, {}, {}) failed: {status:?}",
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
                            "  MISMATCH {label} ({},{},{}): vendor={e:.15e} cintx={a:.15e}",
                            t[0], t[1], t[2]
                        );
                    }
                }
            }
        }

        println!(
            "per-tuple extended-Rys {label}: triples={} elements={compared}",
            list.len()
        );
        assert_eq!(
            mismatches, 0,
            "{mismatches} of {compared} per-tuple extended-Rys {label} elements exceeded \
             max(atol={ATOL:e}, rtol={RTOL:e})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Synthetic class sweep — the arms the real bases cannot reach
// ─────────────────────────────────────────────────────────────────────────────

/// The RI-J gradient lists above top out at `nroots = 6`. This sweep reaches
/// every remaining arm of the inline dispatch — f64 Jacobi, f64 Schmidt, dd
/// Schmidt at order 8, dd Jacobi/Laguerre at 9..12 — through the derivative
/// kernels rather than only through the scalar family's sweep.
///
/// The derivative shape adds one unit of angular momentum, so `nroots =
/// (li + lj + lk + 1) / 2 + 1`; the classes below are chosen so that each order
/// 6..=12 appears exactly once.
const SYNTHETIC_CLASSES: [[i32; 3]; 7] = [
    [3, 3, 4], // nroots 6  — f f | g, the def2-TZVP + def2/J gradient shape
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

/// Every extended Rys order, both derivative families, compared against
/// vendored libcint on a synthetic class per order.
#[test]
fn ext_rys_3c2e_deriv_reaches_every_arm() {
    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let (api, label) = api_for(family);
        let mut orders_covered: BTreeSet<usize> = BTreeSet::new();
        let mut mismatches = 0_usize;
        let mut compared = 0_usize;
        let mut nonzero = 0_usize;

        for ls in SYNTHETIC_CLASSES {
            let nroots = deriv_nroots(ls[0] as usize, ls[1] as usize, ls[2] as usize);
            assert!(
                nroots <= EXTENDED_DEVICE_NROOTS,
                "synthetic class {ls:?} asks for nroots={nroots}, past the extended ceiling"
            );
            assert!(
                ls.iter().all(|&l| l <= i32::from(C2S_LMAX)),
                "synthetic class {ls:?} carries l past the c2s table ceiling {C2S_LMAX}"
            );
            let (atm, bas, env) = synthetic_triple(ls);
            let len: usize = 3 * ls.iter().map(|&l| (2 * l + 1) as usize).product::<usize>();

            let mut expected = vec![0.0_f64; len];
            vendor_eval(family, &mut expected, &[0, 1, 2], &atm, 3, &bas, 3, &env);

            let mut actual = vec![0.0_f64; len];
            // SAFETY: `actual` is sized from the vendor's own AO counts times
            // the three derivative components.
            let status = unsafe {
                eval_raw(
                    api,
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
                "{label} synthetic class {ls:?} (nroots={nroots}) was refused: {status:?}"
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
                            "  MISMATCH {label} l={ls:?} nroots={nroots}: vendor={e:.15e} \
                             cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}"
                        );
                    }
                }
            }
            mismatches += class_mismatches;
            orders_covered.insert(nroots);
            println!(
                "  {label} l={ls:?} nroots={nroots} elements={len} worst |diff|/tol={worst:.3} \
                 mismatches={class_mismatches}"
            );
        }

        assert_eq!(
            orders_covered,
            (BASE_DEVICE_NROOTS + 1..=EXTENDED_DEVICE_NROOTS).collect::<BTreeSet<_>>(),
            "{label}: the sweep must reach every extended Rys order"
        );
        // Agreement alone would be satisfied by both sides returning zero.
        assert!(
            nonzero > 0,
            "{label}: no non-zero output at l >= 5; the c2s transform is zeroing again"
        );
        assert_eq!(
            mismatches, 0,
            "{label}: {mismatches} of {compared} synthetic extended-Rys elements exceeded \
             max(atol={ATOL:e}, rtol={RTOL:e})"
        );
    }
}

/// Above the extended ceiling the family stays fail-closed. `nroots = 13` is
/// where the vendor itself would need quadmath, which this build does not
/// compile, so there is no reference to be compatible with and the right answer
/// is a typed refusal — not a clamp back to a lower order, which is what the
/// launcher's `_` match arm would silently do if a class ever reached it.
#[test]
fn ext_rys_3c2e_deriv_still_refuses_past_the_extended_ceiling() {
    // `(8, 8, 8)` gives `(8 + 8 + 8 + 1) / 2 + 1 = 13`.
    let (atm, bas, env) = synthetic_triple([8, 8, 8]);
    assert_eq!(deriv_nroots(8, 8, 8), EXTENDED_DEVICE_NROOTS + 1);
    let len: usize = 3 * 17 * 17 * 17;
    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let (api, label) = api_for(family);
        let mut actual = vec![0.0_f64; len];
        // SAFETY: `actual` is sized from the AO counts of the synthetic triple.
        let status = unsafe {
            eval_raw(
                api,
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
            "{label}: nroots=13 is past the extended ceiling {EXTENDED_DEVICE_NROOTS} \
             and must be refused"
        );
    }
}
