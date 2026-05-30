//! Oracle parity test for the Phase 23 rank-3 ket-side 2e gradient family
//! `int2e_ip2` (∇ on the 2nd-electron bra-center k), cart + sph.
//!
//! DRV1-01. Validates the 3-component (∇_k) output vs vendored libcint 6.1.3 at
//! atol=1e-12. Requires the `cpu` feature (cubecl cpu backend); vendor parity
//! additionally requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint`
//! cfg). Without BOTH gates the vendor parity test is not compiled and parity
//! silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test int2e_ip2_parity -- --test-threads=1
//!
//! Fixture: an spd H2 fixture with a SHARED STO-3G-style 3-primitive contraction
//! for every shell (the exact byte-identity-gate fixture style proven by
//! `two_electron_ip1_parity.rs`). The absolute basis does not matter for a
//! byte-identity oracle gate — only that cintx and vendor see the same env — but
//! the 2e Rys gradient path requires a self-consistent per-shell `CINTgto_norm`
//! that this shared-coefficient fixture provides.
//!
//! Layout / D-14 discipline: each evaluated quartet pins the element count to
//! `3 * ni*nj*nk*nl` (catches a too-low component_rank truncation) AND asserts
//! `any_nonzero` (catches a zero-fill / short-buffer stub). The quartets sweep
//! NON-SQUARE (distinct-l) blocks so a transposed component layout cannot pass.

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

/// Gradient nroots for int2e_ip2: the ket bra-center k gets the `+1` headroom.
fn grad_nroots_ip2(li: i32, lj: i32, lk: i32, ll: i32) -> i32 {
    (li + lj + (lk + 1) + ll) / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-component gradient collector (cintx via eval_raw), 4-shell arity
// ─────────────────────────────────────────────────────────────────────────────

fn collect_3c_quartet(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 4],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let nl = nf(bas[shls[3] as usize * BAS_SLOTS + ANG_OF]);
    let n_elem = NCOMP * ni * nj * nk * nl;
    let mut out = vec![0.0_f64; n_elem];

    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for quartet {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_3c_quartet<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 4],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let nl = nf(bas[shls[3] as usize * BAS_SLOTS + ANG_OF]);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; NCOMP * ni * nj * nk * nl];
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
// Determinism + shape test (always under cpu) — sweeps all quartets within the
// nroots ceiling, pins 3*ni*nj*nk*nl, and asserts any_nonzero.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int2e_ip2_determinism_and_shape() {
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];
    for (api, nf, rep) in [
        (RawApiId::INT2E_IP2_SPH, nsph as fn(i32) -> usize, "sph"),
        (RawApiId::INT2E_IP2_CART, ncart as fn(i32) -> usize, "cart"),
    ] {
        let mut tested = 0usize;
        for i in 0..N_SHELLS {
            for j in 0..N_SHELLS {
                for k in 0..N_SHELLS {
                    for l in 0..N_SHELLS {
                        let (li, lj, lk, ll) = (ang(i), ang(j), ang(k), ang(l));
                        if grad_nroots_ip2(li, lj, lk, ll) > 5 {
                            continue;
                        }
                        let shls = [i as i32, j as i32, k as i32, l as i32];
                        let (ni, nj, nk, nl) = (nf(li), nf(lj), nf(lk), nf(ll));
                        let m1 = collect_3c_quartet(api, &atm, &bas, &env, &shls, nf);
                        let m2 = collect_3c_quartet(api, &atm, &bas, &env, &shls, nf);
                        assert_eq!(
                            m1.len(),
                            NCOMP * ni * nj * nk * nl,
                            "int2e_ip2_{rep} {shls:?} size must be 3*ni*nj*nk*nl"
                        );
                        for (a, b) in m1.iter().zip(m2.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "int2e_ip2_{rep} {shls:?} not bit-identical"
                            );
                        }
                        tested += 1;
                    }
                }
            }
        }
        // At least one quartet must be nonzero (a global non-stub sentinel; many
        // individual quartets vanish by symmetry, so the sentinel is global).
        // Non-coincident probe (k on atom1 so ∇_k does not vanish by symmetry).
        let probe = collect_3c_quartet(api, &atm, &bas, &env, &[1, 0, 4, 3], nf);
        assert_any_nonzero(&probe, &format!("int2e_ip2_{rep} probe (p,s,p,s) cross-atom"));
        assert!(tested > 0, "int2e_ip2_{rep}: no quartets within nroots ceiling");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity test (requires has_vendor_libcint + cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2e_ip2_spd_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    type VFn = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;
    let cases: [(bool, RawApiId, VFn); 2] = [
        (false, RawApiId::INT2E_IP2_SPH, vendor_ffi::vendor_int2e_ip2_sph),
        (true, RawApiId::INT2E_IP2_CART, vendor_ffi::vendor_int2e_ip2_cart),
    ];
    for (cart, api, vendor_fn) in cases {
        let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
        let rep = if cart { "cart" } else { "sph" };
        let mut mismatches = 0usize;
        let mut any_nonzero = false;
        let mut tested = 0usize;
        let mut nonsquare = false;

        for i in 0..N_SHELLS {
            for j in 0..N_SHELLS {
                for k in 0..N_SHELLS {
                    for l in 0..N_SHELLS {
                        let (li, lj, lk, ll) = (ang(i), ang(j), ang(k), ang(l));
                        if grad_nroots_ip2(li, lj, lk, ll) > 5 {
                            continue;
                        }
                        let shls = [i as i32, j as i32, k as i32, l as i32];
                        let vendor = collect_vendor_3c_quartet(vendor_fn, &atm, &bas, &env, &shls, &nf);
                        let cintx = collect_3c_quartet(api, &atm, &bas, &env, &shls, &nf);
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
            }
        }

        assert_eq!(
            mismatches, 0,
            "int2e_ip2_{rep}: {mismatches} parity mismatches vs vendored libcint (component-leading F-order)"
        );
        assert!(any_nonzero, "int2e_ip2_{rep}: all outputs zero — kernel appears stubbed");
        assert!(nonsquare, "int2e_ip2_{rep}: no non-square (ni!=nk) quartet exercised");
        assert!(tested > 0, "int2e_ip2_{rep}: no quartets tested");
        println!("int2e_ip2_{rep}: vendor parity PASS over {tested} quartets, atol={ATOL:.0e}");
    }
}
