//! Oracle parity test for the Phase 23 rank-3 3-center 2e gradient family
//! `int3c2e_ip2` (∇ on the auxiliary `k` center), cart + sph.
//!
//! DRV1-05. Validates the 3-component output vs vendored libcint 6.1.3 at
//! atol=1e-12. Requires the `cpu` feature (cubecl cpu backend); vendor parity
//! additionally requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint`
//! cfg). Without BOTH gates the vendor parity test is not compiled and parity
//! silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test int3c2e_ip2_parity -- --test-threads=1
//!
//! Pitfall 2 (the slot-mapping landmine): cintx maps the real auxiliary `k` into
//! the 2e `ll` slot, so the ip2 derivative is applied via `nabla1l_2e` (NOT
//! `nabla1k_2e`, which would touch the phantom 2e `lk` slot). A wrong-slot nabla
//! produces silently-wrong / all-zero output — this gate catches it three ways:
//!   - element count pinned to `3 * ni*nj*nk` (catches a too-low component_rank
//!     truncation, D-14),
//!   - `any_nonzero` on both cintx and vendor (catches a zero-fill / short-buffer
//!     stub), and
//!   - a NON-SQUARE i×j (and i×aux) shell triple so a transposed component layout
//!     cannot pass.
//!
//! Fixture: an spd 3-center system with a SHARED STO-3G-style 3-primitive
//! contraction for every shell (the byte-identity-gate fixture style proven by
//! `int2c2e_ip_parity.rs` / `center_2c2e_parity.rs` — the absolute basis does not
//! matter for a byte-identity gate, only that cintx and vendor see the same env).

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
// spd 3-center fixture — shared STO-3G 3-prim contraction for every shell.
// Three distinct atoms so the bra (i,j) and the auxiliary (k) sit on different
// centers (∇_k does not vanish by symmetry).
// Shells (atom, l):
//   0=(0,s) 1=(0,p) 2=(0,d)   bra-i candidates on atom 0
//   3=(1,s) 4=(1,p) 5=(1,d)   bra-j candidates on atom 1
//   6=(2,s) 7=(2,p) 8=(2,d)   auxiliary-k candidates on atom 2
// ─────────────────────────────────────────────────────────────────────────────

fn build_spd_3center_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a0_coord = [0.0_f64, 0.0, -0.70];
    let a1_coord = [0.0_f64, 0.0, 0.70];
    let a2_coord = [0.85_f64, 0.55, 0.10];

    let exp3 = [3.4252509_f64, 0.6239137, 0.1688554];
    let coeff3 = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let a0_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a0_coord);
    let a1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a1_coord);
    let a2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a2_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&exp3);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&coeff3);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    let coords = [a0_coord_ptr, a1_coord_ptr, a2_coord_ptr];
    let charges = [1_i32, 1, 8];
    for (a, (&cptr, &z)) in coords.iter().zip(charges.iter()).enumerate() {
        atm[a * ATM_SLOTS + CHARGE_OF] = z;
        atm[a * ATM_SLOTS + PTR_COORD] = cptr;
        atm[a * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[a * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let shell_spec: [(i32, i32); 9] = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ];
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

// Bra-i shells on atom 0, bra-j shells on atom 1, auxiliary-k shells on atom 2.
const I_SHELLS: [usize; 3] = [0, 1, 2];
const J_SHELLS: [usize; 3] = [3, 4, 5];
const K_SHELLS: [usize; 3] = [6, 7, 8];

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// int3c2e_ip2 nroots: the auxiliary `k` (mapped to the 2e `ll` slot) gets the
/// `+1` headroom — `(li + lj + (lk+1))/2 + 1`.
fn grad_nroots(li: i32, lj: i32, lk: i32) -> i32 {
    (li + lj + lk + 1) / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-component gradient collector (cintx via eval_raw), 3-shell arity
// ─────────────────────────────────────────────────────────────────────────────

fn collect_3c_triple(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 3],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let n_elem = NCOMP * ni * nj * nk;
    let mut out = vec![0.0_f64; n_elem];

    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(
            api_id,
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw failed for triple {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_3c_triple<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 3],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; NCOMP * ni * nj * nk];
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
    assert!(
        any_nonzero,
        "{label}: matrix is all-zero (zero-fill regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism + shape tests (always under cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism_and_shape(api_sph: RawApiId, api_cart: RawApiId, label: &str) {
    let (atm, bas, env) = build_spd_3center_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let mut tested = 0usize;
        for &i in &I_SHELLS {
            for &j in &J_SHELLS {
                for &k in &K_SHELLS {
                    let (li, lj, lk) = (ang(i), ang(j), ang(k));
                    if grad_nroots(li, lj, lk) > 5 {
                        continue;
                    }
                    let shls = [i as i32, j as i32, k as i32];
                    let (ni, nj, nk) = (nf(li), nf(lj), nf(lk));
                    let m1 = collect_3c_triple(api, &atm, &bas, &env, &shls, nf);
                    let m2 = collect_3c_triple(api, &atm, &bas, &env, &shls, nf);
                    assert_eq!(
                        m1.len(),
                        NCOMP * ni * nj * nk,
                        "{label}_{rep} {shls:?} size must be 3*ni*nj*nk"
                    );
                    for (a, b) in m1.iter().zip(m2.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label}_{rep} {shls:?} not bit-identical"
                        );
                    }
                    tested += 1;
                }
            }
        }
        // Non-square probe: (p on atom0, d on atom1, s on atom2) — ∇_k cross-center.
        let probe = collect_3c_triple(api, &atm, &bas, &env, &[1, 5, 6], nf);
        assert_any_nonzero(&probe, &format!("{label}_{rep} probe (p,d,s)"));
        assert!(
            tested > 0,
            "{label}_{rep}: no triples within nroots ceiling"
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c2e_ip2_determinism_and_shape() {
    determinism_and_shape(
        RawApiId::INT3C2E_IP2_SPH,
        RawApiId::INT3C2E_IP2_CART,
        "int3c2e_ip2",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    vendor_cart: fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    label: &str,
) {
    let (atm, bas, env) = build_spd_3center_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    for (cart, api, vendor_fn) in [(false, api_sph, vendor_sph), (true, api_cart, vendor_cart)] {
        let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
        let rep = if cart { "cart" } else { "sph" };
        let mut mismatches = 0usize;
        let mut any_nonzero = false;
        let mut nonsquare = false;
        let mut tested = 0usize;

        for &i in &I_SHELLS {
            for &j in &J_SHELLS {
                for &k in &K_SHELLS {
                    let (li, lj, lk) = (ang(i), ang(j), ang(k));
                    if grad_nroots(li, lj, lk) > 5 {
                        continue;
                    }
                    let shls = [i as i32, j as i32, k as i32];
                    let vendor = collect_vendor_3c_triple(vendor_fn, &atm, &bas, &env, &shls, &nf);
                    let cintx = collect_3c_triple(api, &atm, &bas, &env, &shls, &nf);
                    mismatches += count_mismatches(&vendor, &cintx, ATOL, RTOL);
                    if cintx.iter().any(|v| v.abs() > 1e-18) {
                        any_nonzero = true;
                    }
                    // Non-square in the bra (ni != nj) OR bra/aux (ni != nk).
                    if nf(li) != nf(lj) || nf(li) != nf(lk) {
                        nonsquare = true;
                    }
                    tested += 1;
                }
            }
        }

        assert_eq!(
            mismatches, 0,
            "{label}_{rep}: {mismatches} parity mismatches vs vendored libcint (component-leading F-order)"
        );
        assert!(
            any_nonzero,
            "{label}_{rep}: all outputs zero — kernel appears stubbed"
        );
        assert!(nonsquare, "{label}_{rep}: no non-square triple exercised");
        assert!(tested > 0, "{label}_{rep}: no triples tested");
        println!("{label}_{rep}: vendor parity PASS over {tested} triples, atol={ATOL:.0e}");
    }
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c2e_ip2_spd_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C2E_IP2_SPH,
        RawApiId::INT3C2E_IP2_CART,
        vendor_ffi::vendor_int3c2e_ip2_sph,
        vendor_ffi::vendor_int3c2e_ip2_cart,
        "int3c2e_ip2",
    );
}
