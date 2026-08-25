//! SPIKE 003 — hand-checked-vendor-stride
//! ======================================
//! Recorded artifact: `.planning/spikes/003-hand-checked-vendor-stride/`.
//!
//! Confirms the FULL device block offset formula
//!     out[comp * (ni*nj) + (j*ni + i)]
//! against (Arm 1) a HAND-DERIVED analytic value with no vendor, and (Arm 2) vendored
//! libcint on a non-square block where BOTH ni>1 and nj>1 so the i-fastest orientation
//! is genuinely observable (spike 001 could not see it — every STO-3G non-square block
//! has a unit axis).
//!
//! ── Arm 1 : hand-checked component identity (vendor-FREE) ──
//! For a single normalized s-primitive Gaussian centered at R with gauge origin = 0,
//! the position operator gives  <g_R | r_c | g_R> = R_c * <g_R|g_R> = R_c * S.
//! The scale S is read from cintx's OWN scalar overlap (int1e_ovlp); the *relation*
//! r_block[c] == R_c * S is a non-trivial algebraic invariant that a component swap,
//! a wrong-origin, or a component-interleaved layout would all break. R is chosen by
//! hand, so component 0/1/2 are pinned to x/y/z by an independently-known number.
//!
//! ── Arm 2 : i-fastest orientation, vendor-pinned, across rank tiers ──
//! A p(l=1) x d(l=2) two-center block is 3x6 (cart) / 3x5 (sph) — non-square AND not
//! transpose-symmetric. For every tier r/rr/rrr/rrrr and both paths:
//!   * count_mismatches(vendor, cintx)          == 0   (claimed i-fastest layout right)
//!   * count_mismatches(vendor, cintx_jfastest) >  0   (the OTHER orientation is wrong,
//!     so orientation is truly pinned)
//!     The vendor-free half also asserts cintx != cintx_jfastest (fixture is orientation-
//!     sensitive) so the negative control is meaningful even without vendor.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_003 -- --ignored --nocapture
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test spike_axis_fold_003 -- --ignored --nocapture

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

unsafe fn run(
    api: RawApiId,
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) {
    unsafe {
        eval_raw(api, Some(out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed: {e:?}"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arm 1 fixture: ONE atom, ONE s-primitive (nprim=1, nctr=1), gauge origin = 0.
// ─────────────────────────────────────────────────────────────────────────────
const R_CENTER: [f64; 3] = [0.30, -0.50, 0.70];

fn build_single_s_at(r: [f64; 3]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0_f64; 20]; // gauge origin env[PTR_COMMON_ORIG..+3] left = 0
    debug_assert_eq!(env[PTR_COMMON_ORIG], 0.0);

    let coord_ptr = env.len() as i32;
    env.extend_from_slice(&r);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.push(1.2); // single primitive exponent
    let coeff_ptr = env.len() as i32;
    env.push(1.0); // single contraction coefficient

    let mut atm = vec![0_i32; ATM_SLOTS];
    atm[CHARGE_OF] = 1;
    atm[PTR_COORD] = coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0; // s
    bas[NPRIM_OF] = 1;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = exp_ptr;
    bas[PTR_COEFF] = coeff_ptr;

    (atm, bas, env)
}

#[test]
#[ignore = "spike 003 — run explicitly with --ignored"]
fn spike_003_arm1_hand_checked_component_identity() {
    let (atm, bas, env) = build_single_s_at(R_CENTER);
    let shls = [0_i32, 0];

    // Scalar overlap S = <g_R|g_R> (cintx's own scalar kernel).
    let mut s_block = [0.0_f64; 1];
    unsafe {
        run(
            RawApiId::INT1E_OVLP_CART,
            &mut s_block,
            &shls,
            &atm,
            &bas,
            &env,
        )
    };
    let s = s_block[0];
    assert!(s.abs() > 1e-12, "overlap is zero — degenerate fixture");

    // Position operator r (rank 3), 1x1 block → buffer is exactly [<r_x>, <r_y>, <r_z>].
    let mut r_block = [0.0_f64; 3];
    unsafe {
        run(
            RawApiId::INT1E_R_CART,
            &mut r_block,
            &shls,
            &atm,
            &bas,
            &env,
        )
    };

    println!(
        "\n================ SPIKE 003 Arm 1 : hand-checked component identity ================"
    );
    println!("  R (hand-chosen)   = {R_CENTER:?}");
    println!("  S = <g|g>         = {s:.12e}");
    let axes = ["x", "y", "z"];
    let mut max_rel = 0.0_f64;
    for c in 0..3 {
        let expected = R_CENTER[c] * s; // hand-derived invariant
        let observed = r_block[c];
        let rel = (observed - expected).abs() / expected.abs().max(1e-30);
        max_rel = max_rel.max(rel);
        println!(
            "  comp[{c}] = <{}>  expected R_{}*S = {expected:+.12e}  observed = {observed:+.12e}  rel={rel:.2e}",
            axes[c], axes[c]
        );
        assert!(
            rel < 1e-12,
            "comp[{c}] != R_{}*S — component identity / layout / origin broken",
            axes[c]
        );
    }
    println!(
        "  PASS — comp 0/1/2 == x/y/z * S (component axis pinned, origin=0 honored)  max_rel={max_rel:.2e}\n"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Arm 2 fixture: p (l=1) on center A  x  d (l=2) on center B  → 3x6 cart / 3x5 sph.
// ─────────────────────────────────────────────────────────────────────────────
fn build_p_times_d() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a = [0.0_f64, 0.0, 0.0];
    let b = [0.0_f64, 1.1, 0.6];

    let mut env = vec![0.0_f64; 20]; // gauge origin = 0
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&a);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&b);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);
    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (n, &ptr) in [a_ptr, b_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1; // p (bra)
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2; // d (ket)
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    (atm, bas, env)
}

/// Reinterpret a component-leading buffer as if each per-component block were
/// j-fastest (`i*nj + j`) instead of the claimed i-fastest (`j*ni + i`).
fn to_j_fastest(buf: &[f64], rank: usize, ni: usize, nj: usize) -> Vec<f64> {
    let block = ni * nj;
    let mut out = vec![0.0_f64; buf.len()];
    for c in 0..rank {
        let base = c * block;
        for i in 0..ni {
            for j in 0..nj {
                out[base + i * nj + j] = buf[base + j * ni + i];
            }
        }
    }
    out
}

fn mismatches(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x - **y).abs() > ATOL)
        .count()
}

struct Tier {
    label: &'static str,
    rank: usize,
    cart: RawApiId,
    sph: RawApiId,
}

fn ladder() -> Vec<Tier> {
    vec![
        Tier {
            label: "int1e_r   ",
            rank: 3,
            cart: RawApiId::INT1E_R_CART,
            sph: RawApiId::INT1E_R_SPH,
        },
        Tier {
            label: "int1e_rr  ",
            rank: 9,
            cart: RawApiId::INT1E_RR_CART,
            sph: RawApiId::INT1E_RR_SPH,
        },
        Tier {
            label: "int1e_rrr ",
            rank: 27,
            cart: RawApiId::INT1E_RRR_CART,
            sph: RawApiId::INT1E_RRR_SPH,
        },
        Tier {
            label: "int1e_rrrr",
            rank: 81,
            cart: RawApiId::INT1E_RRRR_CART,
            sph: RawApiId::INT1E_RRRR_SPH,
        },
    ]
}

#[cfg(has_vendor_libcint)]
fn vendor_fns(
    label: &str,
) -> (
    fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
) {
    use cintx_oracle::vendor_ffi as v;
    match label.trim() {
        "int1e_r" => (v::vendor_int1e_r_cart, v::vendor_int1e_r_sph),
        "int1e_rr" => (v::vendor_int1e_rr_cart, v::vendor_int1e_rr_sph),
        "int1e_rrr" => (v::vendor_int1e_rrr_cart, v::vendor_int1e_rrr_sph),
        "int1e_rrrr" => (v::vendor_int1e_rrrr_cart, v::vendor_int1e_rrrr_sph),
        other => panic!("no vendor fn for {other}"),
    }
}

#[test]
#[ignore = "spike 003 — run explicitly with --ignored"]
fn spike_003_arm2_orientation_pinned() {
    let (atm, bas, env) = build_p_times_d();
    let shls = [0_i32, 1];
    #[cfg(has_vendor_libcint)]
    let (natm, nbas) = (
        (atm.len() / ATM_SLOTS) as i32,
        (bas.len() / BAS_SLOTS) as i32,
    );

    println!(
        "\n================ SPIKE 003 Arm 2 : i-fastest orientation (p x d, non-square) ================"
    );
    #[cfg(has_vendor_libcint)]
    println!("vendor: LINKED — orientation pinned against ground truth");
    #[cfg(not(has_vendor_libcint))]
    println!(
        "vendor: NOT linked — negative-control only (cintx != j-fastest); run with CINTX_ORACLE_BUILD_VENDOR=1 to pin"
    );

    for path in ["cart", "sph"] {
        let nf: fn(i32) -> usize = if path == "cart" { ncart } else { nsph };
        let ni = nf(1); // p
        let nj = nf(2); // d
        println!(
            "\n  ---- {path} path : ni={ni} (p) x nj={nj} (d), block={} ----",
            ni * nj
        );
        for t in ladder() {
            let api = if path == "cart" { t.cart } else { t.sph };
            let mut cintx = vec![0.0_f64; t.rank * ni * nj];
            unsafe { run(api, &mut cintx, &shls, &atm, &bas, &env) };
            let cintx_jf = to_j_fastest(&cintx, t.rank, ni, nj);

            // Vendor-free negative control: the two orientations genuinely differ.
            let self_diff = mismatches(&cintx, &cintx_jf);
            assert!(
                self_diff > 0,
                "{} {path}: i-fastest and j-fastest interpretations are identical — \
                 fixture not orientation-sensitive (block transpose-symmetric?)",
                t.label
            );

            #[cfg(has_vendor_libcint)]
            {
                let (vc, vs) = vendor_fns(t.label);
                let vfn = if path == "cart" { vc } else { vs };
                let mut vendor = vec![0.0_f64; t.rank * ni * nj];
                vfn(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);
                let mm_claimed = mismatches(&vendor, &cintx);
                let mm_swapped = mismatches(&vendor, &cintx_jf);
                println!(
                    "    {}  rank={:>2}  mm(vendor,cintx)={mm_claimed:>3}  mm(vendor,j-fastest)={mm_swapped:>3}  self_diff={self_diff:>3}",
                    t.label, t.rank
                );
                assert_eq!(
                    mm_claimed, 0,
                    "{} {path}: cintx diverges from vendor (layout)",
                    t.label
                );
                assert!(
                    mm_swapped > 0,
                    "{} {path}: j-fastest ALSO matches vendor — orientation not actually pinned",
                    t.label
                );
            }
            #[cfg(not(has_vendor_libcint))]
            println!(
                "    {}  rank={:>2}  self_diff={self_diff:>3} (i-fastest vs j-fastest differ)",
                t.label, t.rank
            );
        }
    }
    println!("\n================ SPIKE 003 Arm 2 : done ================\n");
}
