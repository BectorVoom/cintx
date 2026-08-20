//! SPIKE 005 — nctr-axisfold-composition
//! =====================================
//! Recorded artifact: `.planning/spikes/005-nctr-axisfold-composition/`.
//!
//! Spikes 001/003/004 all used nctr==1. The component-leading fold is ASSUMED to compose
//! with general-contraction (nctr>1) blocking, where each axis index is contraction-MAJOR:
//!     i_global = ci*di + ic   (ci = contraction 0..nctr_i, ic = angular 0..di)
//!     out[comp * (ni_full*nj_full) + (j_global*ni_full + i_global)]
//! This is exactly the layout whose ROW/COLUMN-major transpose was a latent historical bug
//! (the nctr>1 coefficient-transpose family). This spike probes the composition across all
//! four rank tiers r/rr/rrr/rrrr.
//!
//! Method:
//!   Vendor-FREE: comp_stride == ni_full*nj_full; clean `rank` slices; the contraction-
//!     order negative control (reinterpreting the i-axis as contraction-MINOR,
//!     i_alt = ic*nctr_i + ci) CHANGES the buffer (i.e. nctr>1 makes the order observable).
//!   Vendor: mm(vendor,cintx)==0 at every tier AND mm(vendor, contraction-minor)>0 — so the
//!     contraction-major composition is the one libcint uses.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_005 -- --ignored --nocapture
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test spike_axis_fold_005 -- --ignored --nocapture

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
fn mismatches(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x - **y).abs() > ATOL)
        .count()
}

/// p-shell (bra, nctr=2) × d-shell (ket, nctr=1), two centers, NON-ZERO gauge origin.
/// Mirrors `moment_genctr_parity.rs::build_moment_genctr_fixture` (the known-good nctr>1
/// moment fixture). Column-major env coeff block is what libcint expects.
fn build_genctr_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80]; // column-major: c0=(.70,.30,.15) c1=(.20,.55,.80)
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let mut env = vec![0.0_f64; 20];
    env[PTR_COMMON_ORIG] = 0.30;
    env[PTR_COMMON_ORIG + 1] = -0.45;
    env[PTR_COMMON_ORIG + 2] = 0.60;
    let i_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);
    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (n, &ptr) in [i_ptr, j_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
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

/// Reinterpret the i-axis of a component-leading, contraction-blocked buffer as
/// contraction-MINOR (`i_alt = ic*nctr_i + ci`) instead of the claimed contraction-MAJOR
/// (`i_claimed = ci*di + ic`). j-axis untouched (ket nctr=1 here).
fn i_axis_contraction_minor(
    buf: &[f64],
    rank: usize,
    nctr_i: usize,
    di: usize,
    nj_full: usize,
) -> Vec<f64> {
    let ni_full = nctr_i * di;
    let block = ni_full * nj_full;
    let mut out = vec![0.0_f64; buf.len()];
    for c in 0..rank {
        let base = c * block;
        for jg in 0..nj_full {
            for ci in 0..nctr_i {
                for ic in 0..di {
                    let i_claimed = ci * di + ic;
                    let i_alt = ic * nctr_i + ci;
                    out[base + jg * ni_full + i_claimed] = buf[base + jg * ni_full + i_alt];
                }
            }
        }
    }
    out
}

struct Tier {
    label: &'static str,
    rank: usize,
    cart: RawApiId,
    sph: RawApiId,
    vcart: &'static str,
    vsph: &'static str,
}

fn ladder() -> Vec<Tier> {
    vec![
        Tier {
            label: "int1e_r   ",
            rank: 3,
            cart: RawApiId::INT1E_R_CART,
            sph: RawApiId::INT1E_R_SPH,
            vcart: "r",
            vsph: "r",
        },
        Tier {
            label: "int1e_rr  ",
            rank: 9,
            cart: RawApiId::INT1E_RR_CART,
            sph: RawApiId::INT1E_RR_SPH,
            vcart: "rr",
            vsph: "rr",
        },
        Tier {
            label: "int1e_rrr ",
            rank: 27,
            cart: RawApiId::INT1E_RRR_CART,
            sph: RawApiId::INT1E_RRR_SPH,
            vcart: "rrr",
            vsph: "rrr",
        },
        Tier {
            label: "int1e_rrrr",
            rank: 81,
            cart: RawApiId::INT1E_RRRR_CART,
            sph: RawApiId::INT1E_RRRR_SPH,
            vcart: "rrrr",
            vsph: "rrrr",
        },
    ]
}

#[cfg(has_vendor_libcint)]
fn vendor_fn(
    key: &str,
    cart: bool,
) -> fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32 {
    use cintx_oracle::vendor_ffi as v;
    match (key, cart) {
        ("r", true) => v::vendor_int1e_r_cart,
        ("r", false) => v::vendor_int1e_r_sph,
        ("rr", true) => v::vendor_int1e_rr_cart,
        ("rr", false) => v::vendor_int1e_rr_sph,
        ("rrr", true) => v::vendor_int1e_rrr_cart,
        ("rrr", false) => v::vendor_int1e_rrr_sph,
        ("rrrr", true) => v::vendor_int1e_rrrr_cart,
        ("rrrr", false) => v::vendor_int1e_rrrr_sph,
        _ => panic!("no vendor fn {key}"),
    }
}

#[test]
#[ignore = "spike 005 — run explicitly with --ignored"]
fn spike_005_nctr_axisfold_composition() {
    let (atm, bas, env) = build_genctr_fixture();
    let shls = [0_i32, 1];
    let nctr_i = bas[NCTR_OF] as usize; // 2
    let li = bas[ANG_OF]; // p
    let lj = bas[BAS_SLOTS + ANG_OF]; // d
    let nctr_j = bas[BAS_SLOTS + NCTR_OF] as usize; // 1
    #[cfg(has_vendor_libcint)]
    let (natm, nbas) = (
        (atm.len() / ATM_SLOTS) as i32,
        (bas.len() / BAS_SLOTS) as i32,
    );

    println!("\n================ SPIKE 005 : nctr>1 × axis-fold composition ================");
    println!(
        "  bra p nctr={nctr_i}, ket d nctr={nctr_j}  (i_global = ci*di + ic, contraction-major)"
    );
    #[cfg(has_vendor_libcint)]
    println!("vendor: LINKED");
    #[cfg(not(has_vendor_libcint))]
    println!("vendor: NOT linked — composition split + order-sensitivity only");

    for rep in ["cart", "sph"] {
        let nf: fn(i32) -> usize = if rep == "cart" { ncart } else { nsph };
        let di = nf(li);
        let dj = nf(lj);
        let ni_full = nctr_i * di;
        let nj_full = nctr_j * dj;
        let block = ni_full * nj_full;
        println!(
            "\n  ---- {rep}: ni_full={ni_full} (={nctr_i}×{di}) nj_full={nj_full} block={block} ----"
        );
        for t in ladder() {
            let api = if rep == "cart" { t.cart } else { t.sph };
            let mut cintx = vec![0.0_f64; t.rank * block];
            unsafe {
                eval_raw(
                    api,
                    Some(&mut cintx),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("{} {rep} nctr>1 eval_raw failed: {e:?}", t.label));
            }
            assert_eq!(
                cintx.len() / t.rank,
                block,
                "{} {rep}: comp_stride != ni_full*nj_full",
                t.label
            );

            // negative control: contraction-minor reinterpretation must differ.
            let minor = i_axis_contraction_minor(&cintx, t.rank, nctr_i, di, nj_full);
            assert!(
                mismatches(&cintx, &minor) > 0,
                "{} {rep}: contraction-minor == contraction-major (nctr structure not observable)",
                t.label
            );

            #[cfg(has_vendor_libcint)]
            {
                let key = if rep == "cart" { t.vcart } else { t.vsph };
                let vfn = vendor_fn(key, rep == "cart");
                let mut vendor = vec![0.0_f64; t.rank * block];
                vfn(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);
                let mm = mismatches(&vendor, &cintx);
                assert_eq!(
                    mm, 0,
                    "{} {rep}: cintx != vendor at nctr>1 (composition broken)",
                    t.label
                );
                let mm_minor = mismatches(&vendor, &minor);
                assert!(
                    mm_minor > 0,
                    "{} {rep}: contraction-MINOR also matches vendor — order not pinned",
                    t.label
                );
                println!(
                    "    {}  rank={:>2}  mm(vendor,cintx)=0  mm(vendor,ctr-minor)={mm_minor}",
                    t.label, t.rank
                );
            }
            #[cfg(not(has_vendor_libcint))]
            println!(
                "    {}  rank={:>2}  comp_stride={block} ✓  ctr-order-sensitive ✓",
                t.label, t.rank
            );
        }
    }
    println!("\n================ SPIKE 005 : done ================\n");
}
