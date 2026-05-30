//! Oracle parity test for the Phase 25 HESS-03 multi-center rank-9 Hessian
//! families: `int2c2e_ipip1` (2-center, ∇² on center 1), `int3c2e_ipip1`
//! (3-center, ∇² on bra center 1), and `int3c2e_ipip2` (3-center, ∇² on center 2
//! — KET headroom), cart + sph.
//!
//! HESS-03 (D-08 register + D-09 transpose discipline). Validates every one of
//! the 9 Hessian components vs vendored libcint 6.1.3 at atol=1e-12. Requires the
//! `cpu` feature (cubecl cpu backend); vendor parity additionally requires
//! `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg). Without BOTH
//! gates the vendor parity test is not compiled and parity silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test hess_multicenter_ipip_parity -- --test-threads=1
//!
//! D-09 ket/bra discipline: `int3c2e_ipip2` raises the KET (auxiliary k → 2e `ll`
//! slot) headroom; the NON-SQUARE block ensures the auxiliary l differs from the
//! bra-pair l so a transposed / bra-confused layout cannot pass. `int2c2e_ipip1`
//! and `int3c2e_ipip1` raise the bra-i headroom.
//!
//! The gate catches three failure modes per D-10/D-09:
//!   - element count pinned to `9 * ni*nj*nk` (3c2e) / `9 * ni*nk` (2c2e) — a
//!     too-low component_rank truncates trailing components,
//!   - `any_nonzero` on cintx output — a zero-fill / short-buffer stub, and
//!   - a NON-SQUARE block (distinct bra/ket l) — a transposed component layout.
//!
//! Fixture: an spd multi-center system with a SHARED STO-3G-style 3-primitive
//! contraction for every shell (the byte-identity-gate fixture style proven by
//! `int3c2e_ip2_parity.rs` / `int2c2e_ip_parity.rs`).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;
const NCOMP: usize = 9;

// ─────────────────────────────────────────────────────────────────────────────
// spd 3-center fixture — shared STO-3G 3-prim contraction for every shell.
// Three distinct atoms so the bra (i,j) and the auxiliary (k) sit on different
// centers (∇²_k does not vanish by symmetry).
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

/// Multi-center Hessian families route through the host `fill_g_tensor_2e` path.
/// The host Rys engine (FND-02) supports nroots 6..12. The headroom per family is
/// `i_inc` (bra-side ipip1) or `k_inc` (ket-side ipip2).
fn hess_nroots_3c(li: i32, lj: i32, lk: i32, i_inc: i32, k_inc: i32) -> i32 {
    (li + i_inc + lj + lk + k_inc) / 2 + 1
}

fn hess_nroots_2c(li: i32, lk: i32, i_inc: i32) -> i32 {
    (li + i_inc + lk) / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collectors via eval_raw (3-shell for 3c2e, 2-shell for 2c2e).
// ─────────────────────────────────────────────────────────────────────────────

fn collect_triple(
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
    let mut out = vec![0.0_f64; NCOMP * ni * nj * nk];
    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for triple {shls:?}: {e:?}"));
    }
    out
}

fn collect_pair(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 2],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let mut out = vec![0.0_f64; NCOMP * ni * nk];
    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for pair {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_triple<F>(
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

#[cfg(has_vendor_libcint)]
fn collect_vendor_pair<F>(
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
            if mismatches <= 20 {
                eprintln!(
                    "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                     diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
    }
    mismatches
}

#[allow(dead_code)]
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: matrix is all-zero (zero-fill regression)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism + shape test (always under cpu) — exercises all 3 families, cart +
// sph, NON-SQUARE blocks, pins 9*ni*nj*nk (3c2e) / 9*ni*nk (2c2e), asserts
// any_nonzero. Catches a too-low component_rank truncation without the vendor gate.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn hess_multicenter_ipip_determinism_and_shape() {
    let (atm, bas, env) = build_spd_3center_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    // ── int2c2e_ipip1 (2-shell, bra-i ∇², i_inc=2) ──
    for (api, nf, rep) in [
        (RawApiId::INT2C2E_IPIP1_SPH, nsph as fn(i32) -> usize, "sph"),
        (RawApiId::INT2C2E_IPIP1_CART, ncart as fn(i32) -> usize, "cart"),
    ] {
        let mut nonsquare = false;
        let mut tested = 0usize;
        for &i in I_SHELLS.iter() {
            for &k in K_SHELLS.iter() {
                let (li, lk) = (ang(i), ang(k));
                if hess_nroots_2c(li, lk, 2) > 12 {
                    continue;
                }
                let shls = [i as i32, k as i32];
                let (ni, nk) = (nf(li), nf(lk));
                let m1 = collect_pair(api, &atm, &bas, &env, &shls, nf);
                let m2 = collect_pair(api, &atm, &bas, &env, &shls, nf);
                assert_eq!(m1.len(), NCOMP * ni * nk, "int2c2e_ipip1_{rep} {shls:?} size");
                for (a, b) in m1.iter().zip(m2.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "int2c2e_ipip1_{rep} {shls:?} not bit-identical");
                }
                if nf(li) != nf(lk) {
                    nonsquare = true;
                }
                tested += 1;
            }
        }
        // p×d cross-center probe — non-vanishing second derivative.
        let probe = collect_pair(api, &atm, &bas, &env, &[1, 8], nf);
        assert_any_nonzero(&probe, &format!("int2c2e_ipip1_{rep} probe (p,d)"));
        assert!(nonsquare, "int2c2e_ipip1_{rep}: no non-square block exercised");
        assert!(tested > 0, "int2c2e_ipip1_{rep}: no pairs within nroots ceiling");
    }

    // ── int3c2e_ipip1 (bra-i ∇², i_inc=2) + int3c2e_ipip2 (ket-k ∇², k_inc=2) ──
    for (cart, sph, i_inc, k_inc, name) in [
        (RawApiId::INT3C2E_IPIP1_CART, RawApiId::INT3C2E_IPIP1_SPH, 2, 0, "int3c2e_ipip1"),
        (RawApiId::INT3C2E_IPIP2_CART, RawApiId::INT3C2E_IPIP2_SPH, 0, 2, "int3c2e_ipip2"),
    ] {
        for (api, nf, rep) in [
            (sph, nsph as fn(i32) -> usize, "sph"),
            (cart, ncart as fn(i32) -> usize, "cart"),
        ] {
            let mut nonsquare = false;
            let mut tested = 0usize;
            for &i in I_SHELLS.iter() {
                for &j in J_SHELLS.iter() {
                    for &k in K_SHELLS.iter() {
                        let (li, lj, lk) = (ang(i), ang(j), ang(k));
                        if hess_nroots_3c(li, lj, lk, i_inc, k_inc) > 12 {
                            continue;
                        }
                        let shls = [i as i32, j as i32, k as i32];
                        let (ni, nj, nk) = (nf(li), nf(lj), nf(lk));
                        let m1 = collect_triple(api, &atm, &bas, &env, &shls, nf);
                        let m2 = collect_triple(api, &atm, &bas, &env, &shls, nf);
                        assert_eq!(m1.len(), NCOMP * ni * nj * nk, "{name}_{rep} {shls:?} size");
                        for (a, b) in m1.iter().zip(m2.iter()) {
                            assert_eq!(a.to_bits(), b.to_bits(), "{name}_{rep} {shls:?} not bit-identical");
                        }
                        // NON-SQUARE: for ipip2 (ket headroom) the aux k l must
                        // differ from the bra i l so a transposed layout cannot pass.
                        if nf(li) != nf(lk) {
                            nonsquare = true;
                        }
                        tested += 1;
                    }
                }
            }
            // p(i),s(j),d(k) cross-center probe — non-square, non-vanishing.
            let probe = collect_triple(api, &atm, &bas, &env, &[1, 3, 8], nf);
            assert_any_nonzero(&probe, &format!("{name}_{rep} probe (p,s,d)"));
            assert!(nonsquare, "{name}_{rep}: no non-square block exercised");
            assert!(tested > 0, "{name}_{rep}: no triples within nroots ceiling");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity test (requires has_vendor_libcint + cpu) — all 3 families, cart +
// sph, NON-SQUARE blocks, every one of the 9 components, atol=1e-12.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn hess_multicenter_ipip() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_spd_3center_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    // ── int2c2e_ipip1 (2-shell, bra-i ∇²) ──
    for (cart, api, vendor_fn) in [
        (
            false,
            RawApiId::INT2C2E_IPIP1_SPH,
            vendor_ffi::vendor_int2c2e_ipip1_sph
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
        (
            true,
            RawApiId::INT2C2E_IPIP1_CART,
            vendor_ffi::vendor_int2c2e_ipip1_cart
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
    ] {
        let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
        let rep = if cart { "cart" } else { "sph" };
        let mut mismatches = 0usize;
        let mut any_nonzero = false;
        let mut nonsquare = false;
        let mut tested = 0usize;
        for &i in I_SHELLS.iter() {
            for &k in K_SHELLS.iter() {
                let (li, lk) = (ang(i), ang(k));
                if hess_nroots_2c(li, lk, 2) > 12 {
                    continue;
                }
                let shls = [i as i32, k as i32];
                let vendor = collect_vendor_pair(vendor_fn, &atm, &bas, &env, &shls, &nf);
                let cintx = collect_pair(api, &atm, &bas, &env, &shls, &nf);
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
        assert_eq!(mismatches, 0, "int2c2e_ipip1_{rep}: {mismatches} parity mismatches vs vendor");
        assert!(any_nonzero, "int2c2e_ipip1_{rep}: all outputs zero — kernel appears stubbed");
        assert!(nonsquare, "int2c2e_ipip1_{rep}: no non-square block exercised");
        assert!(tested > 0, "int2c2e_ipip1_{rep}: no pairs tested");
        println!("int2c2e_ipip1_{rep}: vendor parity PASS over {tested} pairs, atol={ATOL:.0e}");
    }

    // ── int3c2e_ipip1 (bra-i ∇²) + int3c2e_ipip2 (ket-k ∇²) ──
    type VFn3 = fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32;
    let cases: [(&str, i32, i32, [(bool, RawApiId, VFn3); 2]); 2] = [
        (
            "int3c2e_ipip1",
            2,
            0,
            [
                (false, RawApiId::INT3C2E_IPIP1_SPH, vendor_ffi::vendor_int3c2e_ipip1_sph),
                (true, RawApiId::INT3C2E_IPIP1_CART, vendor_ffi::vendor_int3c2e_ipip1_cart),
            ],
        ),
        (
            "int3c2e_ipip2",
            0,
            2,
            [
                (false, RawApiId::INT3C2E_IPIP2_SPH, vendor_ffi::vendor_int3c2e_ipip2_sph),
                (true, RawApiId::INT3C2E_IPIP2_CART, vendor_ffi::vendor_int3c2e_ipip2_cart),
            ],
        ),
    ];

    for (name, i_inc, k_inc, reps) in cases {
        for (cart, api, vendor_fn) in reps {
            let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
            let rep = if cart { "cart" } else { "sph" };
            let mut mismatches = 0usize;
            let mut any_nonzero = false;
            let mut nonsquare = false;
            let mut tested = 0usize;
            for &i in I_SHELLS.iter() {
                for &j in J_SHELLS.iter() {
                    for &k in K_SHELLS.iter() {
                        let (li, lj, lk) = (ang(i), ang(j), ang(k));
                        if hess_nroots_3c(li, lj, lk, i_inc, k_inc) > 12 {
                            continue;
                        }
                        let shls = [i as i32, j as i32, k as i32];
                        let vendor = collect_vendor_triple(vendor_fn, &atm, &bas, &env, &shls, &nf);
                        let cintx = collect_triple(api, &atm, &bas, &env, &shls, &nf);
                        mismatches += count_mismatches(&vendor, &cintx, ATOL, RTOL);
                        if cintx.iter().any(|v| v.abs() > 1e-18) {
                            any_nonzero = true;
                        }
                        // NON-SQUARE: bra i vs auxiliary k angular momenta differ.
                        if nf(li) != nf(lk) {
                            nonsquare = true;
                        }
                        tested += 1;
                    }
                }
            }
            assert_eq!(mismatches, 0, "{name}_{rep}: {mismatches} parity mismatches vs vendor");
            assert!(any_nonzero, "{name}_{rep}: all outputs zero — kernel appears stubbed");
            assert!(nonsquare, "{name}_{rep}: no non-square (ni!=nk) triple exercised");
            assert!(tested > 0, "{name}_{rep}: no triples tested");
            println!("{name}_{rep}: vendor parity PASS over {tested} triples, atol={ATOL:.0e}");
        }
    }
}
