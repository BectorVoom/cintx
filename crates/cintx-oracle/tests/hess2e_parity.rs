//! Oracle parity test for the Phase 25 HESS-02 2e Hessian families:
//! `int2e_ipip1` / `int2e_ipvip1` / `int2e_ip1ip2` (rank 9) and
//! `int2e_ipip1ipip2` (rank 81, 4th-order 2e), cart + sph.
//!
//! HESS-02 (D-07 re-home + fresh register). Validates every component vs
//! vendored libcint 6.1.3 at atol=1e-12. Requires the `cpu` feature (cubecl cpu
//! backend); vendor parity additionally requires `CINTX_ORACLE_BUILD_VENDOR=1`
//! (the `has_vendor_libcint` cfg). Without BOTH gates the vendor parity test is
//! not compiled and parity silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test hess2e_parity -- --test-threads=1
//!
//! Fixture: an spd H2 fixture with a SHARED STO-3G-style 3-primitive contraction
//! for every shell (the exact byte-identity-gate fixture style proven by
//! `int2e_ip2_parity.rs`). The absolute basis does not matter for a byte-identity
//! oracle gate — only that cintx and vendor see the same env.
//!
//! Layout / D-09/D-10 discipline: each evaluated quartet pins the element count to
//! `NCOMP * ni*nj*nk*nl` (catches a too-low component_rank truncation — the rank-81
//! `int2e_ipip1ipip2` MUST emit all 81 components) AND asserts `any_nonzero`
//! (catches a zero-fill / short-buffer stub). The quartets sweep NON-SQUARE
//! (distinct bra/ket l) blocks so a transposed component layout cannot pass —
//! including the rank-81 case (distinct bra and ket angular momenta).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

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

/// Hessian families MUST route through the host `fill_g_tensor_2e` path. The host
/// Rys engine (FND-02) supports nroots 6..12; the family headroom is encoded per
/// family by `i_inc`/`j_inc`/`k_inc` on `(li,lj,lk,ll)`.
fn hess_nroots(li: i32, lj: i32, lk: i32, ll: i32, i_inc: i32, j_inc: i32, k_inc: i32) -> i32 {
    (li + i_inc + lj + j_inc + lk + k_inc + ll) / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// N-component collector (cintx via eval_raw), 4-shell arity, parameterized rank
// ─────────────────────────────────────────────────────────────────────────────

fn collect_quartet(
    api_id: RawApiId,
    ncomp: usize,
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
    let n_elem = ncomp * ni * nj * nk * nl;
    let mut out = vec![0.0_f64; n_elem];

    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for quartet {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_quartet<F>(
    vendor_fn: F,
    ncomp: usize,
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
    let mut out = vec![0.0_f64; ncomp * ni * nj * nk * nl];
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

fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: matrix is all-zero (zero-fill regression)");
}

/// Per-family descriptor for the parametric sweep: name, rank, the cart/sph
/// `RawApiId` pair, and the `(i_inc, j_inc, k_inc)` headroom that drives nroots.
struct HessFamily {
    name: &'static str,
    ncomp: usize,
    cart: RawApiId,
    sph: RawApiId,
    i_inc: i32,
    j_inc: i32,
    k_inc: i32,
}

fn families() -> Vec<HessFamily> {
    vec![
        HessFamily {
            name: "int2e_ipip1",
            ncomp: 9,
            cart: RawApiId::INT2E_IPIP1_CART,
            sph: RawApiId::INT2E_IPIP1_SPH,
            i_inc: 2,
            j_inc: 0,
            k_inc: 0,
        },
        HessFamily {
            name: "int2e_ipvip1",
            ncomp: 9,
            cart: RawApiId::INT2E_IPVIP1_CART,
            sph: RawApiId::INT2E_IPVIP1_SPH,
            i_inc: 1,
            j_inc: 1,
            k_inc: 0,
        },
        HessFamily {
            name: "int2e_ip1ip2",
            ncomp: 9,
            cart: RawApiId::INT2E_IP1IP2_CART,
            sph: RawApiId::INT2E_IP1IP2_SPH,
            i_inc: 1,
            j_inc: 0,
            k_inc: 1,
        },
        HessFamily {
            name: "int2e_ipip1ipip2",
            ncomp: 81,
            cart: RawApiId::INT2E_IPIP1IPIP2_CART,
            sph: RawApiId::INT2E_IPIP1IPIP2_SPH,
            i_inc: 2,
            j_inc: 0,
            k_inc: 2,
        },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism + shape test (always under cpu) — sweeps all quartets within the
// nroots ceiling, pins NCOMP*ni*nj*nk*nl, and asserts any_nonzero. Exercises the
// rank-81 family so a too-low component_rank truncation is caught even without
// the vendor gate.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn hess2e_ipip_determinism_and_shape() {
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];
    for fam in families() {
        for (api, nf, rep) in [
            (fam.sph, nsph as fn(i32) -> usize, "sph"),
            (fam.cart, ncart as fn(i32) -> usize, "cart"),
        ] {
            let mut tested = 0usize;
            let mut nonsquare = false;
            for i in 0..N_SHELLS {
                for j in 0..N_SHELLS {
                    for k in 0..N_SHELLS {
                        for l in 0..N_SHELLS {
                            let (li, lj, lk, ll) = (ang(i), ang(j), ang(k), ang(l));
                            if hess_nroots(li, lj, lk, ll, fam.i_inc, fam.j_inc, fam.k_inc) > 12 {
                                continue;
                            }
                            let shls = [i as i32, j as i32, k as i32, l as i32];
                            let (ni, nj, nk, nl) = (nf(li), nf(lj), nf(lk), nf(ll));
                            let m1 = collect_quartet(api, fam.ncomp, &atm, &bas, &env, &shls, nf);
                            let m2 = collect_quartet(api, fam.ncomp, &atm, &bas, &env, &shls, nf);
                            assert_eq!(
                                m1.len(),
                                fam.ncomp * ni * nj * nk * nl,
                                "{}_{rep} {shls:?} size must be {}*ni*nj*nk*nl",
                                fam.name,
                                fam.ncomp
                            );
                            for (a, b) in m1.iter().zip(m2.iter()) {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{}_{rep} {shls:?} not bit-identical",
                                    fam.name
                                );
                            }
                            if nf(li) != nf(lk) {
                                nonsquare = true;
                            }
                            tested += 1;
                        }
                    }
                }
            }
            // Cross-atom non-coincident probe so the second derivative does not
            // vanish by symmetry (bra p on atom0, ket p on atom1 → non-square,
            // distinct centers).
            let probe = collect_quartet(api, fam.ncomp, &atm, &bas, &env, &[1, 0, 4, 3], nf);
            assert_any_nonzero(&probe, &format!("{}_{rep} probe (p,s,p,s) cross-atom", fam.name));
            assert!(nonsquare, "{}_{rep}: no non-square (ni!=nk) quartet exercised", fam.name);
            assert!(tested > 0, "{}_{rep}: no quartets within nroots ceiling", fam.name);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity test (requires has_vendor_libcint + cpu) — all 4 families, cart +
// sph, NON-SQUARE blocks, every component (ranks 9/9/9/81), atol=1e-12.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn hess2e_ipip() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_spd_fixture();
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];

    type VFn = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;

    // (family, ncomp, i_inc, j_inc, k_inc, [(cart?, RawApiId, vendor_fn); 2])
    struct Case {
        name: &'static str,
        ncomp: usize,
        i_inc: i32,
        j_inc: i32,
        k_inc: i32,
        reps: [(bool, RawApiId, VFn); 2],
    }

    let cases: [Case; 4] = [
        Case {
            name: "int2e_ipip1",
            ncomp: 9,
            i_inc: 2,
            j_inc: 0,
            k_inc: 0,
            reps: [
                (false, RawApiId::INT2E_IPIP1_SPH, vendor_ffi::vendor_int2e_ipip1_sph),
                (true, RawApiId::INT2E_IPIP1_CART, vendor_ffi::vendor_int2e_ipip1_cart),
            ],
        },
        Case {
            name: "int2e_ipvip1",
            ncomp: 9,
            i_inc: 1,
            j_inc: 1,
            k_inc: 0,
            reps: [
                (false, RawApiId::INT2E_IPVIP1_SPH, vendor_ffi::vendor_int2e_ipvip1_sph),
                (true, RawApiId::INT2E_IPVIP1_CART, vendor_ffi::vendor_int2e_ipvip1_cart),
            ],
        },
        Case {
            name: "int2e_ip1ip2",
            ncomp: 9,
            i_inc: 1,
            j_inc: 0,
            k_inc: 1,
            reps: [
                (false, RawApiId::INT2E_IP1IP2_SPH, vendor_ffi::vendor_int2e_ip1ip2_sph),
                (true, RawApiId::INT2E_IP1IP2_CART, vendor_ffi::vendor_int2e_ip1ip2_cart),
            ],
        },
        Case {
            name: "int2e_ipip1ipip2",
            ncomp: 81,
            i_inc: 2,
            j_inc: 0,
            k_inc: 2,
            reps: [
                (false, RawApiId::INT2E_IPIP1IPIP2_SPH, vendor_ffi::vendor_int2e_ipip1ipip2_sph),
                (true, RawApiId::INT2E_IPIP1IPIP2_CART, vendor_ffi::vendor_int2e_ipip1ipip2_cart),
            ],
        },
    ];

    for case in cases {
        for (cart, api, vendor_fn) in case.reps {
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
                            if hess_nroots(li, lj, lk, ll, case.i_inc, case.j_inc, case.k_inc) > 12 {
                                continue;
                            }
                            let shls = [i as i32, j as i32, k as i32, l as i32];
                            let vendor = collect_vendor_quartet(
                                vendor_fn, case.ncomp, &atm, &bas, &env, &shls, &nf,
                            );
                            let cintx =
                                collect_quartet(api, case.ncomp, &atm, &bas, &env, &shls, &nf);
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
                "{}_{rep}: {mismatches} parity mismatches vs vendored libcint (rank {})",
                case.name, case.ncomp
            );
            assert!(any_nonzero, "{}_{rep}: all outputs zero — kernel appears stubbed", case.name);
            assert!(nonsquare, "{}_{rep}: no non-square (ni!=nk) quartet exercised", case.name);
            assert!(tested > 0, "{}_{rep}: no quartets tested", case.name);
            println!(
                "{}_{rep}: vendor parity PASS over {tested} quartets (rank {}), atol={ATOL:.0e}",
                case.name, case.ncomp
            );
        }
    }
}
