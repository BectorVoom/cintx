//! GENERAL-CONTRACTION (`nctr > 1`) vendor parity for the `origi` family —
//! `int1e_r2_origi_sph` / `int1e_r4_origi_sph`.
//!
//! The only pre-existing origi oracle (`origi_random_rocm_parity.rs`) runs on
//! H2O/STO-3G, where every shell is `nctr == 1` AND the single p shell sits on
//! O, so the only p×p pair is the same-shell diagonal block (symmetric under
//! i↔j for an origin-on-i operator). That fixture therefore cannot see either
//! of the two things this file gates:
//!
//!   1. **`nctr > 1`.** libcint emits `[di*nctr_i, dj*nctr_j]` with contraction
//!      as the MAJOR index within each axis (`i_global = ci*di + i_idx`); a
//!      launcher that folds every contraction column into one `di*dj` block
//!      returns silently-wrong values and leaves the rest of the buffer zero.
//!   2. **A non-symmetric same-l pair.** `p(A) × p(B)` on distinct centers
//!      distinguishes the `[j][i]` (i-fastest) block order libcint and
//!      `cart_to_sph_1e` both use from its transpose.
//!
//! Fixture shells (all 3 primitives):
//!   0: p, nctr = 2, atom 0   (general contraction)
//!   1: d, nctr = 1, atom 1
//!   2: s, nctr = 2, atom 1   (general contraction, pure s/p — the GTH-SZV Li shape)
//!   3: p, nctr = 1, atom 1
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
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;

fn build_origi_genctr_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.0_f64, 1.3, 0.7];

    // Shared 3-primitive exponent/coefficient blocks.
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    // libcint env coefficients are COLUMN-MAJOR: env[ci*nprim + ip].
    //   column 0 = (0.70, 0.30, 0.15), column 1 = (0.20, 0.55, 0.80)
    let p_gc_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];
    let p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let s_gc_coeff = [
        0.15432897_f64,
        0.53532814,
        0.44463454,
        0.62_f64,
        -0.31,
        0.18,
    ];

    let mut env = Vec::<f64>::new();
    env.resize(20, 0.0);
    env[PTR_COMMON_ORIG] = 0.30;
    env[PTR_COMMON_ORIG + 1] = -0.45;
    env[PTR_COMMON_ORIG + 2] = 0.60;

    let a_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_coord_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
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

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (n, &ptr) in [a_coord_ptr, b_coord_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    // (atom, l, nprim, nctr, exp_ptr, coeff_ptr)
    let specs: [(i32, i32, i32, i32, i32, i32); 4] = [
        (0, 1, 3, 2, p_exp_ptr, p_gc_coeff_ptr),
        (1, 2, 3, 1, d_exp_ptr, d_coeff_ptr),
        (1, 0, 3, 2, s_exp_ptr, s_gc_coeff_ptr),
        (1, 1, 3, 1, p_exp_ptr, p_coeff_ptr),
    ];
    let mut bas = vec![0_i32; specs.len() * BAS_SLOTS];
    for (s, &(atom, l, nprim, nctr, eptr, cptr)) in specs.iter().enumerate() {
        let b = s * BAS_SLOTS;
        bas[b + ATOM_OF] = atom;
        bas[b + ANG_OF] = l;
        bas[b + NPRIM_OF] = nprim;
        bas[b + NCTR_OF] = nctr;
        bas[b + PTR_EXP] = eptr;
        bas[b + PTR_COEFF] = cptr;
    }

    (atm, bas, env)
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn ao_len(bas: &[i32], s: usize) -> usize {
    nsph(bas[s * BAS_SLOTS + ANG_OF]) * bas[s * BAS_SLOTS + NCTR_OF] as usize
}

/// Every pair worth gating: gc×seg, seg×gc, gc×gc, and the cross-center p×p
/// that a transposed block order would fail. Same-center s×p is deliberately
/// absent — `r^2` about center i makes it vanish by parity, so it gates nothing.
const PAIRS: [[i32; 2]; 8] = [
    [0, 1],
    [1, 0],
    [0, 3],
    [3, 0],
    [0, 2],
    [2, 0],
    [2, 2],
    [3, 3],
];

const OPERATORS: [&str; 2] = ["int1e_r2_origi_sph", "int1e_r4_origi_sph"];

fn eval_cintx(
    symbol: &'static str,
    shls: &[i32; 2],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let n = ao_len(bas, shls[0] as usize) * ao_len(bas, shls[1] as usize);
    let mut out = vec![0.0_f64; n];
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
#[test]
fn test_int1e_origi_genctr_determinism() {
    let (atm, bas, env) = build_origi_genctr_fixture();
    for symbol in OPERATORS {
        for shls in PAIRS {
            let a = eval_cintx(symbol, &shls, &atm, &bas, &env);
            let b = eval_cintx(symbol, &shls, &atm, &bas, &env);
            assert_eq!(
                a.len(),
                ao_len(&bas, shls[0] as usize) * ao_len(&bas, shls[1] as usize),
                "{symbol} {shls:?}: element count = (nsph_i*nctr_i)*(nsph_j*nctr_j)"
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

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_origi_genctr_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_origi_genctr_fixture();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut mismatches = 0usize;
    for symbol in OPERATORS {
        for shls in PAIRS {
            let cintx = eval_cintx(symbol, &shls, &atm, &bas, &env);
            let mut vendor = vec![0.0_f64; cintx.len()];
            match symbol {
                "int1e_r2_origi_sph" => vendor_ffi::vendor_int1e_r2_origi_sph(
                    &mut vendor,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                ),
                "int1e_r4_origi_sph" => vendor_ffi::vendor_int1e_r4_origi_sph(
                    &mut vendor,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                ),
                _ => unreachable!(),
            };
            assert!(
                vendor.iter().any(|v| v.abs() > 1e-14),
                "{symbol} {shls:?}: vendor buffer all-zero — fixture is degenerate"
            );
            for (idx, (&r, &o)) in vendor.iter().zip(cintx.iter()).enumerate() {
                if (o - r).abs() > ATOL {
                    mismatches += 1;
                    eprintln!(
                        "  MISMATCH {symbol} shls {shls:?} idx {idx}: \
                         vendor={r:.15e}, cintx={o:.15e}, diff={:.3e}",
                        (o - r).abs()
                    );
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "origi (nctr>1): {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}
