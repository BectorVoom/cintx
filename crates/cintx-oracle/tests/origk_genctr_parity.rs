//! GENERAL-CONTRACTION (`nctr > 1`) vendor parity for the `origk` family —
//! `int3c1e_r{2,4,6}_origk_sph` and their `ip1` gradients.
//!
//! `origk_random_rocm_parity.rs` / `origk_ip1_random_rocm_parity.rs` both run on
//! H2O/STO-3G, where every shell is `nctr == 1`. The device kernel already emits
//! one Cartesian block per `(ci,cj,ck)`; what no fixture reached was the HOST
//! side, which handed the whole multi-block buffer to `cart_to_sph_3c1e` as if
//! it were a single block. On an all-s/p triple — the GTH-SZV Li shape, where
//! every c2s axis is the identity — that lands in `c2s_apply`'s
//! `out.copy_from_slice(cart)` with mismatched lengths and panics.
//!
//! Fixture shells (all 3 primitives, three distinct centers):
//!   0: p, nctr = 2, atom 0   (general contraction)
//!   1: d, nctr = 1, atom 1
//!   2: s, nctr = 2, atom 2   (general contraction)
//!   3: p, nctr = 1, atom 1
//!   4: s, nctr = 1, atom 2
//!
//! `int3c1e_ip1_r6_origk_sph` is gated here too, and byte-identity for it is
//! deliberate UPSTREAM-BUG PARITY rather than mathematical correctness:
//! `CINTgout1e_int3c1e_ip1_r6_origk` (libcint-master/src/cint3c1e_a.c:627) reads
//! `g76` in the `s[1]` term `6*g48[ix]*g76[iy]*g3[iz]`, but its `G1E_D_I` list
//! (cint3c1e_a.c:604-609) covers g64/g67/g79/g112/g124/g127 and omits
//! `G1E_D_I(g76, g12, ...)`. `g76` sits inside the `MALLOC_INSTACK` span but is
//! never written, so what upstream reads there is whatever the cache allocation
//! holds — zero from a fresh mmap-backed `malloc`, stale numbers from a recycled
//! heap chunk. Result compatibility is this project's primary goal, so
//! `origk_ip1_kernel` reproduces the zero-`g76` behaviour and omits the term;
//! restoring it breaks this gate on the y-gradient component of EVERY triple.
//!
//! Because of that, this operator alone is compared through
//! `vendor_int3c1e_ip1_r6_origk_sph_zeroed_cache`, which hands libcint a zeroed
//! caller-owned cache so `g76` is pinned to 0 and the vendor result becomes a
//! function of its inputs. With the default NULL cache, upstream returns results
//! for identical arguments that differ by ~1e-1 depending on call history, so
//! there is nothing stable to assert against.
//!
//! Double gate: vendor assertions require `--features cpu,unstable-source-api`
//! AND `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg).

#![cfg(any(feature = "cpu", feature = "rocm"))]
// `origi`/`origk` are unstable-source families: without this feature every
// `eval_raw` below returns `source-only symbol ... requires feature
// 'unstable-source-api'`, so the file must cfg out rather than fail. Same
// gate as `orig{i,k}_*_random_rocm_parity.rs`; this file's own header asks
// for `--features cpu,unstable-source-api`.
#![cfg(feature = "unstable-source-api")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;

fn build_origk_genctr_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [
        [0.0_f64, 0.0, 0.0],
        [0.0_f64, 1.3, 0.7],
        [0.9_f64, -0.4, 0.2],
    ];

    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    // libcint env coefficients are COLUMN-MAJOR: env[ci*nprim + ip].
    let p_gc_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];
    let p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let s_exp = [3.1093200_f64, 0.8088610, 0.2443608];
    let s_gc_coeff = [
        0.15432897_f64,
        0.53532814,
        0.44463454,
        0.62_f64,
        -0.31,
        0.18,
    ];
    let s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; 20];

    let mut coord_ptrs = [0_i32; 3];
    for (n, c) in coords.iter().enumerate() {
        coord_ptrs[n] = env.len() as i32;
        env.extend_from_slice(c);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_gc_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_gc_coeff);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_gc_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_gc_coeff);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let mut atm = vec![0_i32; coords.len() * ATM_SLOTS];
    for (n, &ptr) in coord_ptrs.iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    // (atom, l, nctr, exp_ptr, coeff_ptr)
    let specs: [(i32, i32, i32, i32, i32); 6] = [
        (0, 1, 2, p_exp_ptr, p_gc_coeff_ptr),
        (1, 2, 1, d_exp_ptr, d_coeff_ptr),
        (2, 0, 2, s_exp_ptr, s_gc_coeff_ptr),
        (1, 1, 1, p_exp_ptr, p_coeff_ptr),
        (2, 0, 1, s_exp_ptr, s_coeff_ptr),
        (0, 1, 1, p_exp_ptr, p_coeff_ptr),
    ];
    let mut bas = vec![0_i32; specs.len() * BAS_SLOTS];
    for (s, &(atom, l, nctr, eptr, cptr)) in specs.iter().enumerate() {
        let b = s * BAS_SLOTS;
        bas[b + ATOM_OF] = atom;
        bas[b + ANG_OF] = l;
        bas[b + NPRIM_OF] = 3;
        bas[b + NCTR_OF] = nctr;
        bas[b + PTR_EXP] = eptr;
        bas[b + PTR_COEFF] = cptr;
    }

    (atm, bas, env)
}

fn ao_len(bas: &[i32], s: usize) -> usize {
    (2 * bas[s * BAS_SLOTS + ANG_OF] + 1) as usize * bas[s * BAS_SLOTS + NCTR_OF] as usize
}

/// `[0, 3, 4]` is the pure s/p triple with a general-contracted bra — the shape
/// that panicked in `c2s.rs` because every c2s axis there is the identity.
const TRIPLES: [[i32; 3]; 9] = [
    [0, 1, 2],
    [0, 3, 4],
    [0, 3, 2],
    [3, 0, 2],
    [1, 0, 4],
    [3, 1, 4],
    [5, 1, 4],
    [5, 1, 2],
    [0, 1, 4],
];

const SCALAR_OPS: [&str; 3] = [
    "int3c1e_r2_origk_sph",
    "int3c1e_r4_origk_sph",
    "int3c1e_r6_origk_sph",
];

const IP1_OPS: [&str; 3] = [
    "int3c1e_ip1_r2_origk_sph",
    "int3c1e_ip1_r4_origk_sph",
    "int3c1e_ip1_r6_origk_sph",
];

fn elem_count(bas: &[i32], shls: &[i32; 3], ncomp: usize) -> usize {
    ncomp
        * ao_len(bas, shls[0] as usize)
        * ao_len(bas, shls[1] as usize)
        * ao_len(bas, shls[2] as usize)
}

fn eval_cintx(
    symbol: &'static str,
    shls: &[i32; 3],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ncomp: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; elem_count(bas, shls, ncomp)];
    // SAFETY: atm/bas/env well-formed by construction; shls in range; out sized exactly.
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw {symbol} failed for shls {shls:?}: {e:?}"));
    }
    out
}

#[cfg(feature = "cpu")]
fn determinism(ops: &[&'static str], ncomp: usize) {
    let (atm, bas, env) = build_origk_genctr_fixture();
    for &symbol in ops {
        for shls in TRIPLES {
            let a = eval_cintx(symbol, &shls, &atm, &bas, &env, ncomp);
            let b = eval_cintx(symbol, &shls, &atm, &bas, &env, ncomp);
            assert_eq!(
                a.len(),
                elem_count(&bas, &shls, ncomp),
                "{symbol} {shls:?}: element count = ncomp*prod(nsph*nctr)"
            );
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{symbol} {shls:?} must be bit-identical"
                );
            }
            assert!(
                a.iter().any(|v| v.abs() > 1e-14),
                "{symbol} {shls:?}: buffer is all-zero (zero-fill regression)"
            );
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_origk_scalar_genctr_determinism() {
    determinism(&SCALAR_OPS, 1);
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_origk_ip1_genctr_determinism() {
    determinism(&IP1_OPS, 3);
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity(ops: &[&'static str], ncomp: usize) {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_origk_genctr_fixture();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut mismatches = 0usize;
    for &symbol in ops {
        for shls in TRIPLES {
            let cintx = eval_cintx(symbol, &shls, &atm, &bas, &env, ncomp);
            let mut vendor = vec![0.0_f64; cintx.len()];
            let call: fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32 =
                match symbol {
                    "int3c1e_r2_origk_sph" => vendor_ffi::vendor_int3c1e_r2_origk_sph,
                    "int3c1e_r4_origk_sph" => vendor_ffi::vendor_int3c1e_r4_origk_sph,
                    "int3c1e_r6_origk_sph" => vendor_ffi::vendor_int3c1e_r6_origk_sph,
                    "int3c1e_ip1_r2_origk_sph" => vendor_ffi::vendor_int3c1e_ip1_r2_origk_sph,
                    "int3c1e_ip1_r4_origk_sph" => vendor_ffi::vendor_int3c1e_ip1_r4_origk_sph,
                    // Zeroed cache: this operator's result is otherwise not a
                    // function of its inputs — see the module doc and
                    // `vendor_int3c1e_ip1_r6_origk_sph_zeroed_cache`.
                    "int3c1e_ip1_r6_origk_sph" => {
                        vendor_ffi::vendor_int3c1e_ip1_r6_origk_sph_zeroed_cache
                    }
                    _ => unreachable!(),
                };
            call(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);
            assert!(
                vendor.iter().any(|v| v.abs() > 1e-14),
                "{symbol} {shls:?}: vendor buffer all-zero — fixture is degenerate"
            );
            for (idx, (&r, &o)) in vendor.iter().zip(cintx.iter()).enumerate() {
                if (o - r).abs() > ATOL {
                    mismatches += 1;
                    if mismatches <= 40 {
                        eprintln!(
                            "  MISMATCH {symbol} shls {shls:?} idx {idx}: \
                             vendor={r:.15e}, cintx={o:.15e}, diff={:.3e}",
                            (o - r).abs()
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "origk (nctr>1): {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_origk_scalar_genctr_parity() {
    vendor_parity(&SCALAR_OPS, 1);
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_origk_ip1_genctr_parity() {
    vendor_parity(&IP1_OPS, 3);
}
