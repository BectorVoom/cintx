//! Oracle parity test for the Phase 23 rank-3 2-center 2e gradient families
//! `int2c2e_ip1` (∇ on bra center i) and `int2c2e_ip2` (∇ on ket center k),
//! cart + sph.
//!
//! DRV1-04. Validates the 3-component output vs vendored libcint 6.1.3 at
//! atol=1e-12. Requires the `cpu` feature (cubecl cpu backend); vendor parity
//! additionally requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint`
//! cfg). Without BOTH gates the vendor parity test is not compiled and parity
//! silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test int2c2e_ip_parity -- --test-threads=1
//!
//! Fixture: an spd H2 fixture with a SHARED STO-3G-style 3-primitive contraction
//! for every shell (the byte-identity-gate fixture style proven by
//! `two_electron_ip1_parity.rs` / `center_2c2e_parity.rs`).
//!
//! Layout / D-14 discipline: each evaluated pair pins the element count to
//! `3 * ni*nk` (catches a too-low component_rank truncation) AND asserts
//! `any_nonzero` (catches a zero-fill / short-buffer stub). The sweep includes
//! NON-SQUARE (ni != nk) pairs so a transposed component layout cannot pass.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;
const NCOMP: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// spd H2 fixture — shared STO-3G 3-prim contraction for every shell.
// Shells (atom, l): 0=(0,s) 1=(0,p) 2=(0,d) 3=(1,s) 4=(1,p) 5=(1,d)
// ─────────────────────────────────────────────────────────────────────────────

fn build_spd_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let h0_coord = [0.0_f64, 0.0, -0.70];
    let h1_coord = [0.0_f64, 0.0, 0.70];

    let exp3 = [3.4252509_f64, 0.6239137, 0.1688554];
    let coeff3 = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let h0_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h0_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&exp3);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&coeff3);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 1;
    atm[PTR_COORD] = h0_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let shell_spec: [(i32, i32); 6] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    let mut bas = vec![0_i32; shell_spec.len() * BAS_SLOTS];
    for (s, &(atom, l)) in shell_spec.iter().enumerate() {
        bas[s * BAS_SLOTS + ATOM_OF] = atom;
        bas[s * BAS_SLOTS + ANG_OF] = l;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }

    (atm, bas, env)
}

const N_SHELLS: usize = 6;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// int2c2e gradient nroots: the derivative center gets the `+1` headroom. For ip1
/// it is `(li+1 + lk)/2 + 1`, for ip2 `(li + lk+1)/2 + 1` — both equal
/// `(li + lk + 1)/2 + 1`.
fn grad_nroots(li: i32, lk: i32) -> i32 {
    (li + lk + 1) / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-component gradient collector (cintx via eval_raw), 2-shell arity
// ─────────────────────────────────────────────────────────────────────────────

fn collect_3c_pair(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 2],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let n_elem = NCOMP * ni * nk;
    let mut out = vec![0.0_f64; n_elem];

    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for pair {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_3c_pair<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 2],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; NCOMP * ni * nk];
    vendor_fn(&mut out, shls, atm, natm, bas, nbas, env);
    out
}

#[allow(dead_code)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "length mismatch");
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: matrix is all-zero (zero-fill regression)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism + shape tests (always under cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism_and_shape(api_sph: RawApiId, api_cart: RawApiId, label: &str) {
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let mut tested = 0usize;
        for i in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                let (li, lk) = (ang(i), ang(k));
                if grad_nroots(li, lk) > 5 {
                    continue;
                }
                let shls = [i as i32, k as i32];
                let (ni, nk) = (nf(li), nf(lk));
                let m1 = collect_3c_pair(api, &atm, &bas, &env, &shls, nf);
                let m2 = collect_3c_pair(api, &atm, &bas, &env, &shls, nf);
                assert_eq!(m1.len(), NCOMP * ni * nk, "{label}_{rep} {shls:?} size must be 3*ni*nk");
                for (a, b) in m1.iter().zip(m2.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{label}_{rep} {shls:?} not bit-identical");
                }
                tested += 1;
            }
        }
        // Cross-atom probe (∇ does not vanish by symmetry): (p on atom0, s on atom1).
        let probe = collect_3c_pair(api, &atm, &bas, &env, &[1, 3], nf);
        assert_any_nonzero(&probe, &format!("{label}_{rep} probe (p,s) cross-atom"));
        assert!(tested > 0, "{label}_{rep}: no pairs within nroots ceiling");
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int2c2e_ip1_determinism_and_shape() {
    determinism_and_shape(RawApiId::INT2C2E_IP1_SPH, RawApiId::INT2C2E_IP1_CART, "int2c2e_ip1");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int2c2e_ip2_determinism_and_shape() {
    determinism_and_shape(RawApiId::INT2C2E_IP2_SPH, RawApiId::INT2C2E_IP2_CART, "int2c2e_ip2");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    vendor_cart: fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    label: &str,
) {
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    for (cart, api, vendor_fn) in [(false, api_sph, vendor_sph), (true, api_cart, vendor_cart)] {
        let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
        let rep = if cart { "cart" } else { "sph" };
        let mut mismatches = 0usize;
        let mut any_nonzero = false;
        let mut nonsquare = false;
        let mut tested = 0usize;

        for i in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                let (li, lk) = (ang(i), ang(k));
                if grad_nroots(li, lk) > 5 {
                    continue;
                }
                let shls = [i as i32, k as i32];
                let vendor = collect_vendor_3c_pair(vendor_fn, &atm, &bas, &env, &shls, &nf);
                let cintx = collect_3c_pair(api, &atm, &bas, &env, &shls, &nf);
                mismatches += count_mismatches(&vendor, &cintx, ATOL, RTOL);
                if cintx.iter().any(|v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
                if nf(li) != nf(lk) {
                    nonsquare = true;
                }
                tested += 1;
            }
        }

        assert_eq!(
            mismatches, 0,
            "{label}_{rep}: {mismatches} parity mismatches vs vendored libcint (component-leading F-order)"
        );
        assert!(any_nonzero, "{label}_{rep}: all outputs zero — kernel appears stubbed");
        assert!(nonsquare, "{label}_{rep}: no non-square (ni!=nk) pair exercised");
        assert!(tested > 0, "{label}_{rep}: no pairs tested");
        println!("{label}_{rep}: vendor parity PASS over {tested} pairs, atol={ATOL:.0e}");
    }
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2c2e_ip1_spd_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT2C2E_IP1_SPH,
        RawApiId::INT2C2E_IP1_CART,
        vendor_ffi::vendor_int2c2e_ip1_sph,
        vendor_ffi::vendor_int2c2e_ip1_cart,
        "int2c2e_ip1",
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2c2e_ip2_spd_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT2C2E_IP2_SPH,
        RawApiId::INT2C2E_IP2_CART,
        vendor_ffi::vendor_int2c2e_ip2_sph,
        vendor_ffi::vendor_int2c2e_ip2_cart,
        "int2c2e_ip2",
    );
}
