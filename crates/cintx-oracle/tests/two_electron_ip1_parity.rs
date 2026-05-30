//! Oracle parity tests for the `int2e_ip1` two-electron gradient (cart + sph).
//!
//! `int2e_ip1` is the two-electron force `∇_A <ij|kl>` — the single
//! highest-impact term in every analytical HF/DFT/MP2/CCSD gradient (GRAD-07).
//!
//! Compares cintx `eval_raw(INT2E_IP1_{SPH,CART})` output against vendored
//! libcint 6.1.3 (`vendor_int2e_ip1_{sph,cart}`) at the Phase 15 unified
//! tolerance (atol=1e-12, rtol=0.0) for s/p/d shell quartets.
//!
//! LAYOUT VALIDATION (Risk R3): libcint writes its gradient output in
//! **component-leading** F-order — `out[comp * (ni*nj*nk*nl) + n]` for comp in
//! 0..3, with the per-component block `n` walking the AO product i-fastest
//! (`[nl][nk][nj][ni]`, ni fastest) matching pyscf-gto `layout_table.rs`. The
//! cintx kernel transposes its interleaved `gout_ip1` output into this identical
//! layout. The element-for-element comparison against vendor below IS therefore
//! the layout-parity gate: if cintx's `[3, nl, nk, nj, ni]` order does not match
//! libcint's component-leading order, the mismatch count is nonzero and the test
//! fails. There is no separate "layout" assertion — the byte-identity comparison
//! against libcint's own component-leading order subsumes it.
//!
//! nroots ceiling (Risk R2): the li→li+1 raise in the gradient pushes
//! `nroots = (li+1+lj+lk+ll)/2 + 1`. Quartets whose gradient nroots exceeds 5
//! (high-l f/g) are R2-deferred (`UnsupportedApi`); this test restricts the
//! sweep to quartets with gradient nroots ≤ 5 (s/p/d only), skipping the rest.
//!
//! Vendor gating: the real parity tests are double-gated on `--features cpu`
//! AND a vendored libcint build (`CINTX_ORACLE_BUILD_VENDOR=1` → the
//! `has_vendor_libcint` cfg). Without both, the parity tests compile out and
//! only the cintx-side determinism sentinel runs.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Gradient nroots for an `int2e_ip1` quartet: the li→li+1 raise pushes
/// `nroots = (li+1 + lj + lk + ll)/2 + 1`. Quartets with nroots > 5 are
/// R2-deferred (UnsupportedApi) and skipped by the sweep.
fn grad_nroots(li: i32, lj: i32, lk: i32, ll: i32) -> usize {
    (((li + 1) + lj + lk + ll) / 2 + 1) as usize
}

fn matches_with_tol(reference: f64, observed: f64, atol: f64, rtol: f64) -> bool {
    let diff = (observed - reference).abs();
    if diff <= atol {
        return true;
    }
    let denom = reference.abs().max(1.0e-15);
    (diff / denom) <= rtol
}

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "output length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (idx, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        if !matches_with_tol(ref_val, obs_val, atol, rtol) {
            mismatches += 1;
            if mismatches <= 16 {
                let diff = (obs_val - ref_val).abs();
                let rel = diff / ref_val.abs().max(1.0e-15);
                eprintln!(
                    "  MISMATCH[{idx}] ref={ref_val:.15e} obs={obs_val:.15e} diff={diff:.3e} rel={rel:.3e}"
                );
            }
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// s/p/d fixture: two H atoms carrying s, p, and d shells (shared exponents).
//
// Distributing s/p/d across two centers keeps the int2e_ip1 gradient nonzero
// (the force term vanishes for fully degenerate same-center symmetric cases) and
// exercises every angular-momentum combination in the s/p/d envelope. Shells:
//   0: H0 s    1: H0 p    2: H0 d
//   3: H1 s    4: H1 p    5: H1 d
// ─────────────────────────────────────────────────────────────────────────────
fn build_spd_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let h0_coord = [0.0_f64, 0.0, -0.70];
    let h1_coord = [0.0_f64, 0.0, 0.70];

    // STO-3G-style H 1s exponents reused for every shell (the absolute basis
    // does not matter for a byte-identity oracle gate — only that cintx and
    // vendor see the same env).
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

    // 6 shells: (atom, l) = (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
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

/// Drive cintx for one int2e_ip1 quartet via the RAW eval_raw path (the same
/// arity-4 raw arm pyscf-gto's intor.rs uses — independent of Phase 18, D-11).
/// `n_elem` is `3 * ni*nj*nk*nl` (component-leading staging).
fn eval_cintx(api: RawApiId, shls: &[i32; 4], n_elem: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_elem];
    unsafe {
        eval_raw(api, Some(&mut out), None, shls, atm, bas, env, None, None).unwrap_or_else(|e| {
            panic!("eval_raw {api:?} failed for shells {shls:?}: {e:?}")
        });
    }
    out
}

#[cfg(has_vendor_libcint)]
fn eval_vendor_sph(shls: &[i32; 4], n_elem: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;
    let mut out = vec![0.0_f64; n_elem];
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    vendor_ffi::vendor_int2e_ip1_sph(&mut out, shls, atm, natm, bas, nbas, env);
    out
}

#[cfg(has_vendor_libcint)]
fn eval_vendor_cart(shls: &[i32; 4], n_elem: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi;
    let mut out = vec![0.0_f64; n_elem];
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    vendor_ffi::vendor_int2e_ip1_cart(&mut out, shls, atm, natm, bas, nbas, env);
    out
}

/// Sweep all s/p/d quartets with gradient nroots ≤ 5 and assert byte-identity
/// vs vendored libcint. The element-for-element comparison validates the
/// component-leading `[3, nl, nk, nj, ni]` F-order (Risk R3).
#[cfg(has_vendor_libcint)]
fn run_ip1_parity(
    label: &str,
    cart: bool,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let nf = |l: i32| if cart { ncart(l) } else { nsph(l) };
    let api = if cart { RawApiId::INT2E_IP1_CART } else { RawApiId::INT2E_IP1_SPH };

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut tested = 0usize;
    let mut skipped = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                for l_sh in 0..N_SHELLS {
                    let (li, lj, lk, ll) = (ang[i_sh], ang[j_sh], ang[k_sh], ang[l_sh]);
                    // R2: skip quartets whose gradient nroots exceeds the rys_root1..5
                    // ceiling (high-l combinations) — these are UnsupportedApi.
                    if grad_nroots(li, lj, lk, ll) > 5 {
                        skipped += 1;
                        continue;
                    }
                    tested += 1;
                    let n_elem = 3 * nf(li) * nf(lj) * nf(lk) * nf(ll);
                    let shls = [i_sh as i32, j_sh as i32, k_sh as i32, l_sh as i32];

                    let vendor_out = if cart {
                        eval_vendor_cart(&shls, n_elem, atm, bas, env)
                    } else {
                        eval_vendor_sph(&shls, n_elem, atm, bas, env)
                    };
                    let cintx_out = eval_cintx(api, &shls, n_elem, atm, bas, env);

                    // Element-for-element byte-identity == component-leading
                    // F-order layout validation (R3).
                    mismatch_count += count_mismatches(&vendor_out, &cintx_out, ATOL, RTOL);
                    if cintx_out.iter().any(|v| v.abs() > 1.0e-18) {
                        any_nonzero = true;
                    }
                }
            }
        }
    }

    let rep = if cart { "cart" } else { "sph" };
    assert_eq!(
        mismatch_count, 0,
        "{label} int2e_ip1_{rep}: parity mismatches vs vendored libcint (component-leading F-order, R3)"
    );
    assert!(
        any_nonzero,
        "{label} int2e_ip1_{rep}: all outputs zero — kernel appears stubbed (any_nonzero sentinel)"
    );
    println!(
        "{label} int2e_ip1_{rep}: vendor parity PASS over {tested} quartets ({skipped} f/g-deferred), atol={ATOL:.0e}"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_ip1_sph_spd() {
    let (atm, bas, env) = build_spd_fixture();
    run_ip1_parity("s/p/d", false, &atm, &bas, &env);
}

#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_ip1_cart_spd() {
    let (atm, bas, env) = build_spd_fixture();
    run_ip1_parity("s/p/d", true, &atm, &bas, &env);
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism / non-stub sentinel — runs WITHOUT a vendor build (cpu-only).
//
// Guards against a zero-fill regression even when the vendored libcint parity
// gate is not compiled in: a representative s/p/d quartet must produce a
// bit-identical, non-zero 3-component result across two evaluations.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
#[cfg(feature = "cpu")]
fn int2e_ip1_sph_determinism_and_nonzero_sentinel() {
    let (atm, bas, env) = build_spd_fixture();
    // (p, s, p, s) quartet: gradient nroots = (1+1 + 0 + 1 + 0)/2 + 1 = 2 ≤ 5.
    let shls = [1_i32, 3, 4, 0];
    let ang = |s: usize| bas[s * BAS_SLOTS + ANG_OF];
    let n_elem = 3
        * nsph(ang(shls[0] as usize))
        * nsph(ang(shls[1] as usize))
        * nsph(ang(shls[2] as usize))
        * nsph(ang(shls[3] as usize));

    let out1 = eval_cintx(RawApiId::INT2E_IP1_SPH, &shls, n_elem, &atm, &bas, &env);
    let out2 = eval_cintx(RawApiId::INT2E_IP1_SPH, &shls, n_elem, &atm, &bas, &env);

    assert_eq!(out1.len(), out2.len());
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "int2e_ip1 not bit-identical across two evals");
    }
    assert!(
        out1.iter().any(|v| v.abs() > 1.0e-18),
        "int2e_ip1 sentinel quartet is all-zero — kernel appears stubbed"
    );
}
