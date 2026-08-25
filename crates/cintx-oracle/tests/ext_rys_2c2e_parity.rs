//! Phase 33, task 33-03 — `int2c2e` is the third family flipped onto the inline
//! extended Rys path.
//!
//! `int2c2e` is RI's `(P|Q)` metric. Its Rys order is
//! `nroots = (li + lk) / 2 + 1` — a sum over only *two* shells, so it is the
//! last of the four to need the extended path and the first whose high orders
//! are hard to reach from a published basis: `l = 4` on both centres is exactly
//! `nroots = 5`.
//!
//! # Why this file is Cartesian
//!
//! Reaching `nroots >= 6` needs `l >= 5` on at least one centre. The spherical
//! path handles that now — the c2s coefficient table covers `l = 0..=15`, from
//! libcint's own `g_trans_cart2sph[]` — but Cartesian is kept deliberately: it
//! puts no transform between the Rys arm under test and the comparison, so a
//! failure here points at the quadrature rather than at the c2s step that
//! `cintc2s_bra_sph_parity` gates separately.

#![cfg(all(feature = "cpu", feature = "extended-device-rys", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling,
};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Absolute floor. `(i|i)` Cartesian blocks reach O(100), so this is the term
/// that stops binding first; the relative one below is what actually gates.
const ATOL: f64 = 1e-11;

/// Relative tolerance — the dd-vs-f80 floor `rys_nroots_sweep_parity` measured
/// on the roots themselves.
const RTOL: f64 = 1e-9;

/// One class per extended Rys order this family can express with the angular
/// momenta libcint itself supports.
const SYNTHETIC_CLASSES: [[i32; 2]; 4] = [
    [5, 5], // nroots 6
    [6, 6], // nroots 7
    [7, 7], // nroots 8 — the only order whose large-x arm is dd Schmidt
    [8, 8], // nroots 9 — dd Jacobi below the breakpoint, dd Laguerre above
];

/// Highest Rys order reached here. Going further needs `l >= 9`, past libcint's
/// own practical envelope for these fixtures.
const REACHABLE_NROOTS_CEILING: usize = 9;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// Two single-primitive shells on two centres at the requested angular momenta.
fn synthetic_pair(ls: [i32; 2]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 2];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for index in 0..2 {
        atm[index * ATM_SLOTS + CHARGE_OF] = 6;
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for (index, &l) in ls.iter().enumerate() {
        let exp_ptr = env.len() as i32;
        env.push(0.7 + 0.3 * index as f64);
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

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// The precondition, and its complement: `int2c2e` is flipped, the 3c2e
/// derivative set is not, in the same build.
#[test]
fn ext_rys_ceiling_is_raised_for_int2c2e() {
    let backend = cpu_backend();
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int2c2e),
        EXTENDED_DEVICE_NROOTS
    );
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int3c2eDeriv),
        BASE_DEVICE_NROOTS,
        "the 3c2e derivative set has not been flipped and must keep the base ceiling"
    );
}

/// **The gate.** One Cartesian `(l|l)` pair per extended Rys order 6..=9,
/// compared against vendored libcint.
#[test]
fn ext_rys_2c2e_cart_matches_vendor() {
    let mut orders_covered: BTreeSet<usize> = BTreeSet::new();
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;

    for ls in SYNTHETIC_CLASSES {
        let nroots = (ls[0] + ls[1]) as usize / 2 + 1;
        let (atm, bas, env) = synthetic_pair(ls);
        let len = ncart(ls[0]) * ncart(ls[1]);

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2c2e_cart(&mut expected, &[0, 1], &atm, 2, &bas, 2, &env);

        let mut actual = vec![0.0_f64; len];
        let status = unsafe {
            eval_raw(
                RawApiId::Symbol("int2c2e_cart"),
                Some(&mut actual),
                None,
                &[0, 1],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        };
        assert!(
            status.is_ok(),
            "class {ls:?} (nroots={nroots}) was refused: {status:?}"
        );

        let mut worst = 0.0_f64;
        let mut class_mismatches = 0_usize;
        for (e, a) in expected.iter().zip(&actual) {
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
        (BASE_DEVICE_NROOTS + 1..=REACHABLE_NROOTS_CEILING).collect::<BTreeSet<_>>(),
        "the sweep must reach every extended Rys order this family can express"
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} extended-Rys 2c2e elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}
