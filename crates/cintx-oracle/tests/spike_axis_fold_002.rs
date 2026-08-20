//! SPIKE 002 — cart-vs-sph-fold-invariance
//! =======================================
//! Recorded artifact: `.planning/spikes/002-cart-vs-sph-fold-invariance/`.
//!
//! Shows the per-component axis-fold is IDENTICAL on the cartesian and spherical
//! transform paths — the only difference is the per-component block dims
//! (ncart→nsph), and the component axis is never touched by the c2s transform.
//!
//! For each rank tier r/rr/rrr/rrrr on a p×d block, assert:
//!   A. component axis outermost in BOTH paths, same component COUNT (rank) in both:
//!        comp_stride_cart == ncart_i*ncart_j ,  comp_stride_sph == nsph_i*nsph_j
//!   B. fold is path-invariant: for EVERY component c,
//!        cart_to_sph_1e(cart_block[c]) == sph_block[c]   (exact, atol 1e-12)
//!      i.e. the device sph staging is per-component c2s of the device cart staging,
//!      component c → component c, no reordering across the transform.
//!
//! Vendor-free by construction: `cart_to_sph_1e` is the very routine cintx uses, and
//! it is independently vendor-checked by `cintc2s_bra_sph_parity.rs`. Proving
//! `sph == c2s(cart)` per component shows the two DEVICE paths fold the component axis
//! the same way.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_002 -- --ignored --nocapture

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_cubecl::transform::c2s::cart_to_sph_1e;

const ATOL: f64 = 1e-12;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// p (l=1) on center A  x  d (l=2) on center B  — the first non-trivial sph transform.
fn build_p_times_d() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a = [0.0_f64, 0.0, 0.0];
    let b = [0.0_f64, 1.1, 0.6];
    let mut env = vec![0.0_f64; 20];
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
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;
    (atm, bas, env)
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

#[test]
#[ignore = "spike 002 — run explicitly with --ignored"]
fn spike_002_cart_vs_sph_fold_invariance() {
    let (atm, bas, env) = build_p_times_d();
    let shls = [0_i32, 1];
    let (li, lj) = (1u8, 2u8); // p × d
    let nci = ncart(li as i32);
    let ncj = ncart(lj as i32);
    let nsi = nsph(li as i32);
    let nsj = nsph(lj as i32);
    let block_cart = nci * ncj;
    let block_sph = nsi * nsj;

    println!(
        "\n================ SPIKE 002 : cart↔sph per-component fold invariance (p×d) ================"
    );
    println!("  cart block = {nci}×{ncj} = {block_cart}   sph block = {nsi}×{nsj} = {block_sph}");

    for t in ladder() {
        let mut cart = vec![0.0_f64; t.rank * block_cart];
        let mut sph = vec![0.0_f64; t.rank * block_sph];
        // SAFETY: fixture well-formed, shls valid.
        unsafe {
            eval_raw(
                t.cart,
                Some(&mut cart),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
            eval_raw(
                t.sph,
                Some(&mut sph),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        // A. component axis outermost + same count in both paths.
        assert_eq!(
            cart.len(),
            t.rank * block_cart,
            "{}: cart comp_stride != ncart_i*ncart_j",
            t.label
        );
        assert_eq!(
            sph.len(),
            t.rank * block_sph,
            "{}: sph comp_stride != nsph_i*nsph_j",
            t.label
        );
        assert_eq!(cart.len() / t.rank, block_cart);
        assert_eq!(sph.len() / t.rank, block_sph);

        // B. per-component fold invariance: sph_block[c] == c2s(cart_block[c]).
        let mut worst = 0.0_f64;
        for c in 0..t.rank {
            let cart_block = &cart[c * block_cart..(c + 1) * block_cart];
            let sph_block = &sph[c * block_sph..(c + 1) * block_sph];
            let mut recon = vec![0.0_f64; block_sph];
            cart_to_sph_1e::<f64>(cart_block, &mut recon, li, lj);
            for (k, (&r, &s)) in recon.iter().zip(sph_block.iter()).enumerate() {
                let d = (r - s).abs();
                worst = worst.max(d);
                assert!(
                    d <= ATOL,
                    "{}: comp {c}/{} elem {k}: c2s(cart)={r:.15e} != sph={s:.15e} (Δ={d:.3e}) \
                     — cart/sph paths fold the component axis differently",
                    t.label,
                    t.rank
                );
            }
        }
        println!(
            "  {}  rank={:>2}  cart_len={:>4} sph_len={:>4}  per-comp c2s(cart)==sph ✓  worst Δ={worst:.2e}",
            t.label,
            t.rank,
            cart.len(),
            sph.len()
        );
    }
    println!(
        "\n================ SPIKE 002 : PASS — fold path-invariant, component axis untouched ================\n"
    );
}
