//! `def2_speed_precision_plan.md` D1.2 — the one-electron **derivative** set is
//! flipped onto the inline extended Rys path.
//!
//! # What this closes
//!
//! `RysFamily::Int1eDeriv` covers six device kernels — the nuclear gradient
//! (`one_electron_nuc_grad_kernel`), `rinv`/`drinv`, the both-side and bra-side
//! second derivatives, and the GIAO nuclear engine. All six stopped at
//! `rys_root5`, and five separate per-tuple guards spelled a literal
//! `MAX_DEVICE_NROOTS` rather than asking the family for its ceiling. Above that
//! they returned a typed `UnsupportedApi` — correct, but a refusal.
//!
//! One of those refusals is not hypothetical for def2-TZVP. The GIAO nuclear
//! engine's shape carries `nmax = li + lj + 5`, so
//! `nroots = (li + lj + 5) / 2 + 1` reaches **6 at `l = 3`** — an `(f|f)` pair,
//! which def2-TZVP has on oxygen. Every other family in this set needs `l >= 4`
//! and so is out of reach of any def2 orbital basis; the GIAO one was being
//! refused on a basis the project targets. That asymmetry is what D1.2 removes,
//! and it is worth stating plainly because the rest of this file is reached only
//! through synthetic high-`l` classes.
//!
//! # Why this file is Cartesian
//!
//! Same reason as `ext_rys_1e_parity` and `ext_rys_2c2e_parity`: reaching
//! `nroots >= 6` needs high `l`, and the Cartesian route keeps the c2s step out
//! of the comparison so a failure points at the quadrature.
//!
//! # Why this file is feature-gated rather than always compiled
//!
//! `extended-device-rys` is off by default and, even when on, effective only
//! where `device_rys_ceiling::fma_fusion_verified` passes for the backend. A
//! test that ran without the feature would be asserting the *old* behaviour.

#![cfg(all(feature = "cpu", feature = "extended-device-rys", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA,
    RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling,
};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use std::collections::BTreeSet;

/// Absolute floor; the relative term below is what binds at large magnitudes.
/// Same pair every extended-Rys gate uses — they share one solver.
const ATOL: f64 = 1e-11;

/// Relative tolerance — the dd-vs-f80 floor `rys_nroots_sweep_parity` measured
/// on the roots themselves.
const RTOL: f64 = 1e-9;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// Two shells on two centres, plus a third nucleus off-axis so the nuclear
/// attraction sums over more than one centre and a wrong root would not cancel.
///
/// `env[PTR_RINV_ORIG]` and `env[PTR_COMMON_ORIG]` are both set to a non-zero,
/// off-centre point: `rinv`/`drinv` read the first, and the GIAO gauge families
/// read the second. A zero gauge origin makes several GIAO gout combinations
/// vanish, which would make agreement meaningless.
fn synthetic_pair(ls: [i32; 2]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3], [0.6, 0.8, 0.5]];
    let charges = [6, 8, 1];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    env[PTR_COMMON_ORIG] = 0.2;
    env[PTR_COMMON_ORIG + 1] = -0.3;
    env[PTR_COMMON_ORIG + 2] = 0.45;
    env[PTR_RINV_ORIG] = 0.15;
    env[PTR_RINV_ORIG + 1] = 0.25;
    env[PTR_RINV_ORIG + 2] = 0.7;

    let mut coord_ptr = [0_i32; 3];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for index in 0..3 {
        atm[index * ATM_SLOTS + CHARGE_OF] = charges[index];
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

/// The precondition: with the feature compiled in, the CPU probe passing and the
/// 1e derivative set now on the flipped list, the ceiling really is raised for
/// this family. Without this every sweep below would pass trivially by being
/// refused and never compared.
#[test]
fn ext_rys_ceiling_is_raised_for_the_1e_derivatives() {
    let backend = cpu_backend();
    assert_eq!(
        device_nroots_ceiling(&backend, RysFamily::Int1eDeriv),
        EXTENDED_DEVICE_NROOTS
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Real-valued families — one sweep helper, five kernels
// ─────────────────────────────────────────────────────────────────────────────

type VendorCart = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;

/// Compare one real-valued 1e derivative family against vendored libcint over a
/// list of `(class, expected nroots)` pairs, and assert the orders actually
/// reached are the ones the caller named.
///
/// `nroots` is passed in rather than recomputed here because each family has its
/// own headroom — `(li + lj) / 2 + 1` for the gradient, `+ 2` for the second
/// derivatives, `+ 5` for the GIAO engine — and hardcoding one formula would
/// quietly mislabel four of the five sweeps.
fn sweep_real(
    label: &str,
    api: RawApiId,
    vendor: VendorCart,
    rank: usize,
    classes: &[([i32; 2], usize)],
) {
    let mut orders: BTreeSet<usize> = BTreeSet::new();
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;
    let mut nonzero = 0_usize;

    for &(ls, nroots) in classes {
        assert!(
            nroots > BASE_DEVICE_NROOTS && nroots <= EXTENDED_DEVICE_NROOTS,
            "{label} class {ls:?} claims nroots={nroots}, outside the extended band"
        );
        let (atm, bas, env) = synthetic_pair(ls);
        let len = rank * ncart(ls[0]) * ncart(ls[1]);

        let mut expected = vec![0.0_f64; len];
        vendor(&mut expected, &[0, 1], &atm, 3, &bas, 2, &env);

        let mut actual = vec![0.0_f64; len];
        // SAFETY: `actual` is sized `rank * ncart(li) * ncart(lj)`, the extent
        // the vendor writes for the same shells.
        let status = unsafe {
            eval_raw(
                api,
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
            "{label} class {ls:?} (nroots={nroots}) was refused: {status:?}"
        );

        let mut worst = 0.0_f64;
        let mut class_mismatches = 0_usize;
        for (e, a) in expected.iter().zip(&actual) {
            compared += 1;
            if e.abs() > 1e-14 {
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
        orders.insert(nroots);
        println!(
            "  {label} l={ls:?} nroots={nroots} elements={len} worst |diff|/tol={worst:.3} \
             mismatches={class_mismatches}"
        );
    }

    // Agreement alone would be satisfied by both sides returning zero, which is
    // exactly what a zero-filled high-`l` path used to produce.
    assert!(
        nonzero > 0,
        "{label}: the vendor reference is all-zero, so agreement proves nothing"
    );
    assert_eq!(
        orders.len(),
        classes.len(),
        "{label}: the sweep must reach a distinct order per class, got {orders:?}"
    );
    assert_eq!(
        mismatches, 0,
        "{label}: {mismatches} of {compared} extended-Rys elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// **`one_electron_nuc_grad_kernel`** — `int1e_ipnuc`, `nroots = (li + lj) / 2 + 1`
/// rounded up. The order stops at 9 for the same reason `ext_rys_1e_parity`
/// stops there: going further needs `l >= 9` on both centres.
#[test]
fn ext_rys_1e_ipnuc_cart_matches_vendor() {
    sweep_real(
        "int1e_ipnuc",
        RawApiId::Symbol("int1e_ipnuc_cart"),
        vendor_ffi::vendor_int1e_ipnuc_cart,
        3,
        &[([5, 5], 6), ([6, 6], 7), ([7, 7], 8), ([8, 8], 9)],
    );
}

/// **`one_electron_rinv_kernel`** — `int1e_rinv`, `nroots = (li + lj) / 2 + 1`.
///
/// This kernel belongs to `RysFamily::Int1e` rather than `Int1eDeriv` (it is a
/// scalar one-electron integral), but it is one of the six kernels this change
/// gave an extended arm, and `ext_rys_1e_parity` covers only
/// `one_electron_scalar_kernel`. Without this sweep the arm would ship untested.
#[test]
fn ext_rys_1e_rinv_cart_matches_vendor() {
    sweep_real(
        "int1e_rinv",
        RawApiId::Symbol("int1e_rinv_cart"),
        vendor_ffi::vendor_int1e_rinv_cart,
        1,
        &[([5, 5], 6), ([6, 6], 7), ([7, 7], 8), ([8, 8], 9)],
    );
}

/// **`one_electron_drinv_kernel`** — `int1e_drinv`,
/// `nroots = (li + lj + 2) / 2 + 1`, so it reaches one order higher than `rinv`
/// at the same `l`.
#[test]
fn ext_rys_1e_drinv_cart_matches_vendor() {
    sweep_real(
        "int1e_drinv",
        RawApiId::Symbol("int1e_drinv_cart"),
        vendor_ffi::vendor_int1e_drinv_cart,
        3,
        &[
            ([4, 4], 6),
            ([5, 5], 7),
            ([6, 6], 8),
            ([7, 7], 9),
            ([8, 8], 10),
        ],
    );
}

/// **`one_electron_nuc_grad_both_kernel`** — `int1e_ipnucip`, the both-side
/// second derivative, `nroots = (li + lj + 2) / 2 + 1`.
#[test]
fn ext_rys_1e_ipnucip_cart_matches_vendor() {
    sweep_real(
        "int1e_ipnucip",
        RawApiId::Symbol("int1e_ipnucip_cart"),
        vendor_ffi::vendor_int1e_ipnucip_cart,
        9,
        &[
            ([4, 4], 6),
            ([5, 5], 7),
            ([6, 6], 8),
            ([7, 7], 9),
            ([8, 8], 10),
        ],
    );
}

/// **`one_electron_nuc_gradgrad_bra_kernel`** — `int1e_ipipnuc`, the bra-side
/// second derivative, same order formula as the both-side one.
#[test]
fn ext_rys_1e_ipipnuc_cart_matches_vendor() {
    sweep_real(
        "int1e_ipipnuc",
        RawApiId::Symbol("int1e_ipipnuc_cart"),
        vendor_ffi::vendor_int1e_ipipnuc_cart,
        9,
        &[
            ([4, 4], 6),
            ([5, 5], 7),
            ([6, 6], 8),
            ([7, 7], 9),
            ([8, 8], 10),
        ],
    );
}

/// **`one_electron_nuc_gradgrad_bra_kernel`, rinv arm** — `int1e_ipiprinv`
/// shares the kernel with `ipipnuc` but takes the single rinv origin and charge
/// `+1` instead of the nuclear sum, so it exercises a different argument path
/// through the same extended arm.
#[test]
fn ext_rys_1e_ipiprinv_cart_matches_vendor() {
    sweep_real(
        "int1e_ipiprinv",
        RawApiId::Symbol("int1e_ipiprinv_cart"),
        vendor_ffi::vendor_int1e_ipiprinv_cart,
        9,
        &[
            ([4, 4], 6),
            ([5, 5], 7),
            ([6, 6], 8),
            ([7, 7], 9),
            ([8, 8], 10),
        ],
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  The GIAO nuclear engine — purely imaginary, so a complex comparison
// ─────────────────────────────────────────────────────────────────────────────

/// **`one_electron_giao_nuc_kernel`** — `int1e_ignuc`,
/// `nroots = (li + lj + 5) / 2 + 1`.
///
/// The GIAO integrals are purely imaginary: libcint returns the real magnitude
/// of the imaginary part as a plain `double*`, while cintx registers the family
/// `complex_output=true` and hands back a `2x`-interleaved `[re, im]` buffer. So
/// the comparison is cintx's odd (imaginary) half against the vendor's output,
/// with cintx's even (real) half required to be exactly zero — the same
/// reconciliation `giao_1e_parity` documents as D-15.
///
/// This is the one family in the set whose order-6 class is `l = 3`, i.e. an
/// `(f|f)` pair that def2-TZVP actually contains. `[3, 3]` therefore leads the
/// sweep and is the concrete def2-TZVP coverage this flip buys.
#[test]
fn ext_rys_1e_giao_nuc_cart_matches_vendor() {
    let classes: [([i32; 2], usize); 6] = [
        ([3, 3], 6), // the def2-TZVP (f|f) class
        ([4, 4], 7),
        ([5, 5], 8),
        ([6, 6], 9),
        ([7, 7], 10),
        ([8, 8], 11),
    ];
    const RANK: usize = 3;

    let mut orders: BTreeSet<usize> = BTreeSet::new();
    let mut mismatches = 0_usize;
    let mut compared = 0_usize;
    let mut nonzero = 0_usize;

    for (ls, nroots) in classes {
        assert_eq!(
            (ls[0] + ls[1] + 5) as usize / 2 + 1,
            nroots,
            "class {ls:?} order table is stale"
        );
        let (atm, bas, env) = synthetic_pair(ls);
        let n = RANK * ncart(ls[0]) * ncart(ls[1]);

        let mut expected = vec![0.0_f64; n];
        vendor_ffi::vendor_int1e_ignuc_cart(&mut expected, &[0, 1], &atm, 3, &bas, 2, &env);

        // Complex output: `2 * n` interleaved [re, im].
        let mut actual = vec![0.0_f64; 2 * n];
        // SAFETY: the family carries `complex_output=true`, so the raw buffer is
        // twice the real extent the vendor writes.
        let status = unsafe {
            eval_raw(
                RawApiId::Symbol("int1e_ignuc_cart"),
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
            "int1e_ignuc class {ls:?} (nroots={nroots}) was refused: {status:?}"
        );

        let mut worst = 0.0_f64;
        let mut class_mismatches = 0_usize;
        for (index, e) in expected.iter().enumerate() {
            compared += 1;
            if e.abs() > 1e-14 {
                nonzero += 1;
            }
            assert_eq!(
                actual[2 * index],
                0.0,
                "int1e_ignuc {ls:?}: real half must be exactly zero at element {index}"
            );
            let a = actual[2 * index + 1];
            let diff = (e - a).abs();
            let tol = ATOL.max(RTOL * e.abs());
            worst = worst.max(diff / tol);
            if diff > tol {
                class_mismatches += 1;
                if class_mismatches <= 5 {
                    eprintln!(
                        "  MISMATCH int1e_ignuc l={ls:?} nroots={nroots}: vendor={e:.15e} \
                         cintx={a:.15e} |d|={diff:.3e} tol={tol:.3e}"
                    );
                }
            }
        }
        mismatches += class_mismatches;
        orders.insert(nroots);
        println!(
            "  int1e_ignuc l={ls:?} nroots={nroots} elements={n} worst |diff|/tol={worst:.3} \
             mismatches={class_mismatches}"
        );
    }

    assert!(
        nonzero > 0,
        "int1e_ignuc: the vendor reference is all-zero, so agreement proves nothing"
    );
    assert_eq!(orders.len(), classes.len(), "orders reached: {orders:?}");
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {compared} extended-Rys int1e_ignuc elements exceeded \
         max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// Above the extended ceiling the family stays fail-closed. `nroots = 13` is
/// where the vendor itself would need quadmath, so there is no reference to be
/// compatible with and the right answer is a typed refusal — not a clamp back to
/// a lower order, which is what a launcher's catch-all arm would silently do.
///
/// `int1e_ignuc` is the family that can reach 13 at the lowest `l`:
/// `(l + l + 5) / 2 + 1 = 13` needs `l = 10`.
#[test]
fn ext_rys_1e_deriv_still_refuses_past_the_extended_ceiling() {
    let ls = [10, 10];
    assert_eq!(
        (ls[0] + ls[1] + 5) as usize / 2 + 1,
        EXTENDED_DEVICE_NROOTS + 1
    );
    let (atm, bas, env) = synthetic_pair(ls);
    let n = 3 * ncart(ls[0]) * ncart(ls[1]);
    let mut actual = vec![0.0_f64; 2 * n];
    // SAFETY: the buffer is twice the real extent for this complex family.
    let status = unsafe {
        eval_raw(
            RawApiId::Symbol("int1e_ignuc_cart"),
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
        status.is_err(),
        "nroots=13 is past the extended ceiling {EXTENDED_DEVICE_NROOTS} and must be refused"
    );
}
