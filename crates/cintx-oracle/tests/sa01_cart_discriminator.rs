//! 30-01c-DEBUG "DECISIVE NEXT EXPERIMENT" — the cart-path discriminator.
//!
//! Splits the rank-9 sa01 byte-identity blocker (`int1e_cg_sa10sa01_spinor` not
//! byte-identical to vendor) into g-tensor-vs-transform by comparing cintx's
//! PRE-transform cart `gc` buffer against the vendored libcint CART driver
//! `int1e_cg_sa10sa01_cart` — which bypasses the c2s_si spinor transform entirely.
//!
//!   cart buffers DIFFER  → bug is in the g-tensor assembly (sa01_g1_bothside /
//!                          sa01_x1i_of_g1 / HRR headroom).
//!   cart buffers MATCH    → bug is in the transform feed/call (re-open si_2d).
//!
//! The comparison is SCALE-INVARIANT: cintx's `gc` carries the s/p sph angular
//! norm (`common_fac_sp`) while the vendor cart driver does not, so absolute
//! magnitudes need not match. What MUST match (if the g-tensor is correct) is the
//! zero/non-zero structure and the relative ratios between the 36 components.
//!
//! Double-gate: real vendor bodies need BOTH `--features cpu` AND
//! `CINTX_ORACLE_BUILD_VENDOR=1`:
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!     --test sa01_cart_discriminator -- --nocapture

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COMMON_ORIG,
    PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG,
};

/// Minimal single-primitive, single-contraction (li, lj) shell-pair fixture with
/// a non-zero gauge origin (`common_orig`) and rinv center pinned to [0,0,0]
/// (matching the cg_sa10sa01 dispatch arm: rinv = PTR_RINV_ORIG default [0,0,0]).
#[cfg(feature = "cpu")]
struct CartFixture {
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
    ri: [f64; 3],
    rj: [f64; 3],
    common_orig: [f64; 3],
    ai: f64,
    aj: f64,
    ci: f64,
    cj: f64,
}

#[cfg(feature = "cpu")]
fn build_cart_fixture(li: u8, lj: u8) -> CartFixture {
    let ri = [0.10_f64, 0.20, 0.30];
    let rj = [0.00_f64, 0.05, 0.90];
    let common_orig = [0.05_f64, -0.10, 0.15];
    let (ai, aj) = (1.3_f64, 0.7_f64);
    let (ci, cj) = (1.0_f64, 1.0_f64);

    // env: [0..PTR_ENV_START) reserved globals, then coords/exps/coeffs.
    let mut env = vec![0.0_f64; PTR_ENV_START];
    env[PTR_COMMON_ORIG] = common_orig[0];
    env[PTR_COMMON_ORIG + 1] = common_orig[1];
    env[PTR_COMMON_ORIG + 2] = common_orig[2];
    // rinv center = [0,0,0] (explicit; matches cg_sa10sa01 arm).
    env[PTR_RINV_ORIG] = 0.0;
    env[PTR_RINV_ORIG + 1] = 0.0;
    env[PTR_RINV_ORIG + 2] = 0.0;

    let coord_i_ptr = env.len();
    env.extend_from_slice(&ri);
    let coord_j_ptr = env.len();
    env.extend_from_slice(&rj);
    let exp_i_ptr = env.len();
    env.push(ai);
    let exp_j_ptr = env.len();
    env.push(aj);
    let coeff_i_ptr = env.len();
    env.push(ci);
    let coeff_j_ptr = env.len();
    env.push(cj);

    // Two atoms (so bra and ket sit at distinct coords).
    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[ATOM_OF] = 0; // charge slot unused by cart sa01 (rinv center is env-driven)
    atm[PTR_COORD] = coord_i_ptr as i32;
    atm[ATM_SLOTS + PTR_COORD] = coord_j_ptr as i32;

    // Two shells: bra (shell 0) at atom 0, ket (shell 1) at atom 1.
    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = li as i32;
    bas[NPRIM_OF] = 1;
    bas[NCTR_OF] = 1;
    bas[KAPPA_OF] = 0; // cart driver ignores kappa
    bas[PTR_EXP] = exp_i_ptr as i32;
    bas[PTR_COEFF] = coeff_i_ptr as i32;

    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = lj as i32;
    bas[BAS_SLOTS + NPRIM_OF] = 1;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = 0;
    bas[BAS_SLOTS + PTR_EXP] = exp_j_ptr as i32;
    bas[BAS_SLOTS + PTR_COEFF] = coeff_j_ptr as i32;

    CartFixture { atm, bas, env, ri, rj, common_orig, ai, aj, ci, cj }
}

#[cfg(feature = "cpu")]
fn ncart(l: u8) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// cintx pre-transform cart gc for the cg_sa10sa01 arm (origin = dri = ri − common_orig).
#[cfg(feature = "cpu")]
fn collect_cintx_cart_gc(fx: &CartFixture, li: u8, lj: u8) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::sa01_cart_gc_for_test;
    use cintx_runtime::{BackendIntent, BackendKind};

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend must initialise");

    let dri = [
        fx.ri[0] - fx.common_orig[0],
        fx.ri[1] - fx.common_orig[1],
        fx.ri[2] - fx.common_orig[2],
    ];

    sa01_cart_gc_for_test(
        &backend,
        li,
        lj,
        1,
        1,
        1,
        1,
        fx.ri,
        fx.rj,
        [0.0, 0.0, 0.0], // rinv center
        dri,             // x1i gauge origin (cg)
        &[fx.ai],
        &[fx.aj],
        &[fx.ci],
        &[fx.cj],
    )
    .expect("sa01 cart gc must build")
}

/// Reorder cintx gc into vendor cart flat order: out[comp*ncj*nci + (j*nci + i)],
/// comp = grp*4 + block (grp 0..9, block x,y,z,1).
///
/// cintx gc layout (nctr=1): gc[(grp*N_GC + block)*block_len + (cj_idx*nci + ci_idx)]
/// with block_len = nci*ncj. For nctr=1 the contraction base is 0. The inner
/// (cj_idx*nci + ci_idx) elem index already equals (j*nci + i) in cintx's own cart
/// ordering — so the two layouts coincide IF cintx and libcint share cart-component
/// ordering (true for s; verified separately for p). N_GC = 4.
#[cfg(feature = "cpu")]
fn cintx_gc_in_vendor_order(gc: &[f64], li: u8, lj: u8) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let n_gc = 4usize;
    let mut out = vec![0.0_f64; 36 * block_len];
    for grp in 0..9 {
        for block in 0..n_gc {
            let comp = grp * n_gc + block;
            for elem in 0..block_len {
                let src = (grp * n_gc + block) * block_len + elem;
                out[comp * block_len + elem] = gc[src];
            }
        }
    }
    out
}

#[cfg(all(feature = "cpu", has_vendor_libcint))]
fn collect_vendor_cart(fx: &CartFixture, li: u8, lj: u8) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::vendor_int1e_cg_sa10sa01_cart;
    let natm = (fx.atm.len() / ATM_SLOTS) as i32;
    let nbas = (fx.bas.len() / BAS_SLOTS) as i32;
    let shls: [i32; 2] = [0, 1];
    let mut out = vec![0.0_f64; 36 * ncart(li) * ncart(lj)];
    vendor_int1e_cg_sa10sa01_cart(&mut out, &shls, &fx.atm, natm, &fx.bas, nbas, &fx.env);
    out
}

/// Print + compare two 36×block_len cart vectors scale-invariantly. Returns the
/// list of component indices whose zero/non-zero structure DIFFERS (the decisive
/// localization signal). Also reports ratio constancy among shared non-zeros.
#[cfg(all(feature = "cpu", has_vendor_libcint))]
fn diagnose(cintx: &[f64], vendor: &[f64], block_len: usize, label: &str) -> Vec<usize> {
    assert_eq!(cintx.len(), vendor.len(), "{label}: length mismatch");
    let zthr = 1e-12;
    let cmax = cintx.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
    let vmax = vendor.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);

    eprintln!("\n===== {label} (36 comp × block_len {block_len}) =====");
    eprintln!("  cintx |max| = {cmax:.6e}   vendor |max| = {vmax:.6e}");
    eprintln!(
        "  {:>4} {:>3} {:>3} | {:>14} {:>14} | {:>10} {:>10} | {}",
        "comp", "grp", "blk", "cintx", "vendor", "cintx/cmax", "vend/vmax", "verdict"
    );

    let mut struct_diffs = Vec::new();
    for comp in 0..36 {
        let grp = comp / 4;
        let blk = comp % 4; // 0=x 1=y 2=z 3=1
        for elem in 0..block_len {
            let idx = comp * block_len + elem;
            let cv = cintx[idx];
            let vv = vendor[idx];
            let cz = cv.abs() <= zthr;
            let vz = vv.abs() <= zthr;
            let verdict = if cz != vz {
                struct_diffs.push(idx);
                "<<< STRUCT DIFF"
            } else if !cz && !vz {
                ""
            } else {
                "(both 0)"
            };
            // Only print rows that are interesting (any non-zero or a struct diff).
            if !cz || !vz {
                eprintln!(
                    "  {comp:>4} {grp:>3} {blk:>3} | {cv:>14.6e} {vv:>14.6e} | {:>10.5} {:>10.5} | {verdict}",
                    cv / cmax,
                    vv / vmax,
                );
            }
        }
    }

    // Ratio constancy among components non-zero in BOTH (normalization-invariant).
    let mut ratios = Vec::new();
    for idx in 0..cintx.len() {
        if cintx[idx].abs() > zthr && vendor[idx].abs() > zthr {
            ratios.push((cintx[idx] / cmax) / (vendor[idx] / vmax));
        }
    }
    if !ratios.is_empty() {
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let maxdev = ratios.iter().fold(0.0_f64, |m, r| m.max((r - mean).abs()));
        eprintln!(
            "  shared-nonzero normalized ratio: mean={mean:.6}  max|dev|={maxdev:.3e}  (n={})",
            ratios.len()
        );
        eprintln!(
            "  → {}",
            if maxdev < 1e-9 {
                "ratios CONSTANT ⇒ g-tensor structurally CORRECT (look at the transform)"
            } else {
                "ratios VARY ⇒ g-tensor assembly is WRONG (per-component mismatch above)"
            }
        );
    }
    eprintln!("  struct diffs (cintx 0 xor vendor 0): {struct_diffs:?}");
    struct_diffs
}

/// s×s (li=lj=0): the cleanest cut — single cart component each, ZERO cart-ordering
/// ambiguity, 36 vs 36 direct. Decides g-tensor-vs-transform on its own.
#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_cart_discriminator_ss() {
    let fx = build_cart_fixture(0, 0);
    let gc = collect_cintx_cart_gc(&fx, 0, 0);
    let cintx = cintx_gc_in_vendor_order(&gc, 0, 0);
    let vendor = collect_vendor_cart(&fx, 0, 0);

    let diffs = diagnose(&cintx, &vendor, ncart(0) * ncart(0), "s×s cg_sa10sa01_cart");

    // Vendor's 6 hard-zero gout slots: comps 1,2,16,18,32,33 (block-1 of grp0/4/8
    // pattern). Any struct diff OUTSIDE those is a real g-tensor zero/non-zero defect.
    let vendor_hard_zeros = [1usize, 2, 16, 18, 32, 33];
    let unexpected: Vec<usize> =
        diffs.iter().copied().filter(|d| !vendor_hard_zeros.contains(d)).collect();
    assert!(
        unexpected.is_empty(),
        "s×s: cintx cart gc has zero/non-zero STRUCTURE differing from vendor at comps {unexpected:?} \
         — g-tensor assembly bug (a contribution is structurally absent or spurious)"
    );
}

/// p×s (li=1, lj=0): surfaces bra-stepping in the x1i / nabla_i recurrences. 3 cart
/// comps on the bra; relies on cintx p-cart ordering == libcint (x,y,z).
#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_cart_discriminator_ps() {
    let fx = build_cart_fixture(1, 0);
    let gc = collect_cintx_cart_gc(&fx, 1, 0);
    let cintx = cintx_gc_in_vendor_order(&gc, 1, 0);
    let vendor = collect_vendor_cart(&fx, 1, 0);
    let _ = diagnose(&cintx, &vendor, ncart(1) * ncart(0), "p×s cg_sa10sa01_cart");
    // No hard assert here (cart-ordering caveat); the printed table is the evidence.
}

/// p×d (li=1, lj=2): the EXACT angular momenta of the failing spinor fixture
/// (p bra, d ket). Exercises both bra p-ordering and ket d-ordering of the cart
/// blocks. If THIS matches vendor cart, the g-tensor is exonerated for the failing
/// geometry and the bug is conclusively in the c2s_si transform path.
#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_cart_discriminator_pd() {
    let fx = build_cart_fixture(1, 2);
    let gc = collect_cintx_cart_gc(&fx, 1, 2);
    let cintx = cintx_gc_in_vendor_order(&gc, 1, 2);
    let vendor = collect_vendor_cart(&fx, 1, 2);
    let diffs = diagnose(&cintx, &vendor, ncart(1) * ncart(2), "p×d cg_sa10sa01_cart");
    // p×d shares libcint cart ordering (bra x,y,z; ket xx,xy,xz,yy,yz,zz), so a
    // constant ratio + no struct diffs is expected if the g-tensor is correct.
    assert!(
        diffs.is_empty(),
        "p×d: cintx cart gc zero/non-zero structure differs from vendor at flat idx {diffs:?} \
         — would indicate a g-tensor bug specific to the failing geometry"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Spinor-pattern probe on the ACTUAL failing fixture (p kappa+1 nctr=2 × d kappa−1).
// Prints the exact cintx-vs-vendor relationship per spinor element so the residual
// transform/scatter bug reveals itself (i-rotation? sign? conj? structural zero?).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
fn spinor_len_kappa(l: i32, kappa: i32) -> usize {
    if kappa == 0 {
        (4 * l + 2) as usize
    } else if kappa < 0 {
        (2 * l + 2) as usize
    } else {
        (2 * l) as usize
    }
}

#[cfg(feature = "cpu")]
struct Shell {
    l: u8,
    kappa: i16,
    nprim: usize,
    nctr: usize,
    coord: [f64; 3],
    exps: Vec<f64>,
    coeff_row_major: Vec<f64>,
}

#[cfg(feature = "cpu")]
fn extract_shell(s: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Shell {
    let l = bas[s * BAS_SLOTS + ANG_OF] as u8;
    let kappa = bas[s * BAS_SLOTS + KAPPA_OF] as i16;
    let nprim = bas[s * BAS_SLOTS + NPRIM_OF] as usize;
    let nctr = bas[s * BAS_SLOTS + NCTR_OF] as usize;
    let atom = bas[s * BAS_SLOTS + ATOM_OF] as usize;
    let exp_ptr = bas[s * BAS_SLOTS + PTR_EXP] as usize;
    let coeff_ptr = bas[s * BAS_SLOTS + PTR_COEFF] as usize;
    let coord_ptr = atm[atom * ATM_SLOTS + PTR_COORD] as usize;
    let coord = [env[coord_ptr], env[coord_ptr + 1], env[coord_ptr + 2]];
    let exps = env[exp_ptr..exp_ptr + nprim].to_vec();
    let col_major = &env[coeff_ptr..coeff_ptr + nprim * nctr];
    let mut coeff_row_major = vec![0.0_f64; nprim * nctr];
    for ic in 0..nctr {
        for ip in 0..nprim {
            coeff_row_major[ip * nctr + ic] = col_major[ic * nprim + ip];
        }
    }
    Shell { l, kappa, nprim, nctr, coord, exps, coeff_row_major }
}

/// PINPOINT: for nprim=1 general contraction, the nctr=2 cart gc's per-ci block must
/// equal a nctr=1 run with that contraction's coeff (linearity of contraction). If the
/// ci-blocks are NOT proportional to their coeffs, the rank-9 kernel's contraction loop
/// is broken. Uses sa01_cart_gc_for_test directly (no vendor needed).
#[cfg(feature = "cpu")]
#[test]
fn sa01_nctr2_gc_linearity() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::sa01_cart_gc_for_test;
    use cintx_runtime::{BackendIntent, BackendKind};

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");

    let (li, lj) = (1u8, 2u8);
    let ri = [0.10_f64, 0.20, 0.30];
    let rj = [0.00_f64, 0.05, 0.90];
    let origin = [0.05_f64, -0.10, 0.15];
    let (ai, aj) = (1.3_f64, 0.7_f64);
    let (c0, c1) = (0.9_f64, 1.7_f64); // two distinct contraction coeffs for bra
    let cj0 = 1.1_f64;

    // nprim=1, nctr_i=2 (row-major coeff [c0, c1]); nctr_j=1.
    let gc2 = sa01_cart_gc_for_test(
        &backend, li, lj, 1, 1, 2, 1, ri, rj, [0.0; 3], origin, &[ai], &[aj], &[c0, c1], &[cj0],
    )
    .expect("nctr=2 gc");
    // Reference: two separate nctr=1 runs, one per contraction coeff.
    let gc_c0 = sa01_cart_gc_for_test(
        &backend, li, lj, 1, 1, 1, 1, ri, rj, [0.0; 3], origin, &[ai], &[aj], &[c0], &[cj0],
    )
    .expect("nctr=1 c0");
    let gc_c1 = sa01_cart_gc_for_test(
        &backend, li, lj, 1, 1, 1, 1, ri, rj, [0.0; 3], origin, &[ai], &[aj], &[c1], &[cj0],
    )
    .expect("nctr=1 c1");

    let total_len = 9 * 4 * ncart(li) * ncart(lj); // per (ci,cj) block
    assert_eq!(gc2.len(), 2 * total_len, "nctr=2 gc has 2 contraction blocks");
    assert_eq!(gc_c0.len(), total_len);

    let blk0 = &gc2[0..total_len]; // ci=0
    let blk1 = &gc2[total_len..2 * total_len]; // ci=1
    let dev0 = blk0.iter().zip(gc_c0.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let dev1 = blk1.iter().zip(gc_c1.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("\n=== sa01 nctr=2 gc linearity ===");
    eprintln!("  ci=0 block vs nctr1(c0): max dev = {dev0:.3e}");
    eprintln!("  ci=1 block vs nctr1(c1): max dev = {dev1:.3e}");
    eprintln!("  |ci=0 block|max = {:.4e}  |nctr1(c0)|max = {:.4e}",
        blk0.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        gc_c0.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
    eprintln!("  |ci=1 block|max = {:.4e}  |nctr1(c1)|max = {:.4e}",
        blk1.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        gc_c1.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
    assert!(dev0 < 1e-12, "ci=0 block diverges from nctr=1(c0) by {dev0:.3e} — kernel contraction bug");
    assert!(dev1 < 1e-12, "ci=1 block diverges from nctr=1(c1) by {dev1:.3e} — kernel contraction bug");
}

/// PINPOINT #2: nprim=3, nctr=2 (the REAL fixture's bra shell shape). For general
/// contraction the primitives are shared; each ci block must still equal a nctr=1 run
/// with that contraction's 3-primitive coeff column. Tests multi-primitive × nctr>1.
#[cfg(feature = "cpu")]
#[test]
fn sa01_nprim3_nctr2_gc_linearity() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::sa01_cart_gc_for_test;
    use cintx_runtime::{BackendIntent, BackendKind};

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");

    let (li, lj) = (1u8, 2u8);
    let ri = [0.10_f64, 0.20, 0.30];
    let rj = [0.00_f64, 0.05, 0.90];
    let origin = [0.05_f64, -0.10, 0.15];
    // Real-fixture bra: 3 primitives, 2 general-contraction columns.
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    // row-major [ip*nctr+ic]: col0=(0.70,0.30,0.15) col1=(0.20,0.55,0.80)
    let p_coeff_rm = [0.70_f64, 0.20, 0.30, 0.55, 0.15, 0.80];
    let col0 = [0.70_f64, 0.30, 0.15];
    let col1 = [0.20_f64, 0.55, 0.80];
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let gc2 = sa01_cart_gc_for_test(
        &backend, li, lj, 3, 3, 2, 1, ri, rj, [0.0; 3], origin, &p_exp, &d_exp, &p_coeff_rm,
        &d_coeff,
    )
    .expect("nprim3 nctr2");
    let gc_c0 = sa01_cart_gc_for_test(
        &backend, li, lj, 3, 3, 1, 1, ri, rj, [0.0; 3], origin, &p_exp, &d_exp, &col0, &d_coeff,
    )
    .expect("nctr1 col0");
    let gc_c1 = sa01_cart_gc_for_test(
        &backend, li, lj, 3, 3, 1, 1, ri, rj, [0.0; 3], origin, &p_exp, &d_exp, &col1, &d_coeff,
    )
    .expect("nctr1 col1");

    let total_len = 9 * 4 * ncart(li) * ncart(lj);
    let blk0 = &gc2[0..total_len];
    let blk1 = &gc2[total_len..2 * total_len];
    let dev0 = blk0.iter().zip(gc_c0.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let dev1 = blk1.iter().zip(gc_c1.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("\n=== sa01 nprim=3 nctr=2 gc linearity ===");
    eprintln!("  ci=0 block vs nctr1(col0): max dev = {dev0:.3e}");
    eprintln!("  ci=1 block vs nctr1(col1): max dev = {dev1:.3e}");
    assert!(dev0 < 1e-12, "ci=0 block diverges by {dev0:.3e}");
    assert!(dev1 < 1e-12, "ci=1 block diverges by {dev1:.3e}");
}

/// Build a p(kappa_i)×d(kappa_j) SPINOR fixture, single contraction (nctr=1), with a
/// non-zero gauge origin — same geometry as build_cart_fixture but with real kappa so
/// the spinor transform runs. li/lj fixed at 1/2 (p×d).
#[cfg(all(feature = "cpu", has_vendor_libcint))]
fn build_spinor_nctr1_fixture(kappa_i: i32, kappa_j: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let fx = build_cart_fixture(1, 2);
    let mut bas = fx.bas;
    bas[KAPPA_OF] = kappa_i;
    bas[BAS_SLOTS + KAPPA_OF] = kappa_j;
    (fx.atm, bas, fx.env)
}

/// Build p(kappa_i, nctr=2)×d(kappa_j) SPINOR fixture, single primitive, gauge≠0.
/// Isolates nctr=2 in the spinor launcher with controlled simple data.
#[cfg(all(feature = "cpu", has_vendor_libcint))]
fn build_spinor_nctr2_fixture(kappa_i: i32, kappa_j: i32, c0: f64, c1: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let ri = [0.10_f64, 0.20, 0.30];
    let rj = [0.00_f64, 0.05, 0.90];
    let common_orig = [0.05_f64, -0.10, 0.15];
    let (ai, aj) = (1.3_f64, 0.7_f64);

    let mut env = vec![0.0_f64; PTR_ENV_START];
    env[PTR_COMMON_ORIG] = common_orig[0];
    env[PTR_COMMON_ORIG + 1] = common_orig[1];
    env[PTR_COMMON_ORIG + 2] = common_orig[2];
    let coord_i_ptr = env.len();
    env.extend_from_slice(&ri);
    let coord_j_ptr = env.len();
    env.extend_from_slice(&rj);
    let exp_i_ptr = env.len();
    env.push(ai);
    let exp_j_ptr = env.len();
    env.push(aj);
    // bra p, nprim=1, nctr=2: COLUMN-major env coeff [c0, c1] (1 prim each).
    let coeff_i_ptr = env.len();
    env.push(c0);
    env.push(c1);
    let coeff_j_ptr = env.len();
    env.push(1.1);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[PTR_COORD] = coord_i_ptr as i32;
    atm[ATM_SLOTS + PTR_COORD] = coord_j_ptr as i32;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 1;
    bas[NCTR_OF] = 2;
    bas[KAPPA_OF] = kappa_i;
    bas[PTR_EXP] = exp_i_ptr as i32;
    bas[PTR_COEFF] = coeff_i_ptr as i32;
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 1;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = kappa_j;
    bas[BAS_SLOTS + PTR_EXP] = exp_j_ptr as i32;
    bas[BAS_SLOTS + PTR_COEFF] = coeff_j_ptr as i32;
    (atm, bas, env)
}

#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_spinor_nctr2_controlled() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_oracle::vendor_ffi::{vendor_CINTcgto_spinor, vendor_int1e_cg_sa10sa01_spinor};
    use cintx_runtime::{BackendIntent, BackendKind};

    let (atm, bas, env) = build_spinor_nctr2_fixture(1, -1, 0.9, 1.7);
    let si = extract_shell(0, &atm, &bas, &env);
    let sj = extract_shell(1, &atm, &bas, &env);
    let di = spinor_len_kappa(1, 1);
    let dj = spinor_len_kappa(2, -1);
    let ni_sp = si.nctr * di; // 2*2 = 4
    let nj_sp = sj.nctr * dj; // 6

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");
    let common_orig = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    let rinv_orig = [env[PTR_RINV_ORIG], env[PTR_RINV_ORIG + 1], env[PTR_RINV_ORIG + 2]];
    let mut cintx = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend, "cg_sa10sa01", si.l, si.kappa, sj.l, sj.kappa, si.nprim, sj.nprim, si.nctr,
        sj.nctr, si.coord, sj.coord, common_orig, rinv_orig, &si.exps, &sj.exps, &si.coeff_row_major,
        &sj.coeff_row_major, &[], &[], &mut cintx,
    )
    .expect("cintx nctr2 controlled");

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    assert_eq!(vendor_CINTcgto_spinor(0, &bas) as usize, ni_sp);
    let shls = [0_i32, 1];
    let mut vendor = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    vendor_int1e_cg_sa10sa01_spinor(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

    let mism = count_mismatches(&cintx, &vendor, 1e-9, 0.0);
    eprintln!("\n=== sa01 SPINOR nctr=2 CONTROLLED (nprim=1): mismatch = {mism}/{} ===", ni_sp * nj_sp * 9 * 2);
    // per-group half-zero tally
    let group_out = ni_sp * nj_sp * 2;
    for grp in 0..9 {
        let (mut both, mut half) = (0, 0);
        for k in 0..group_out / 2 {
            let idx = grp * group_out + k * 2;
            let rz = cintx[idx].abs() <= 1e-12;
            let iz = cintx[idx + 1].abs() <= 1e-12;
            if !rz && !iz { both += 1; } else if rz != iz { half += 1; }
        }
        eprintln!("  group {grp}: both={both} half-zero={half}");
    }
    assert_eq!(mism, 0, "nctr=2 controlled spinor must match vendor");
}

#[cfg(all(feature = "cpu", has_vendor_libcint))]
fn count_mismatches(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| (**x - **y).abs() > atol + rtol * y.abs()).count()
}

/// DISCRIMINATOR: does the half-zero spinor bug persist at nctr=1? If yes → core
/// transform bug (nctr-independent). If it vanishes → general-contraction (gc/scatter)
/// bug. Compares cintx vs vendor rank-9 spinor on p(kappa+1,nctr=1)×d(kappa−1).
#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_spinor_nctr1_pattern() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_oracle::vendor_ffi::{vendor_CINTcgto_spinor, vendor_int1e_cg_sa10sa01_spinor};
    use cintx_runtime::{BackendIntent, BackendKind};

    let (atm, bas, env) = build_spinor_nctr1_fixture(1, -1); // p LT × d GT
    let si = extract_shell(0, &atm, &bas, &env);
    let sj = extract_shell(1, &atm, &bas, &env);
    let di = spinor_len_kappa(si.l as i32, si.kappa as i32); // 2
    let dj = spinor_len_kappa(sj.l as i32, sj.kappa as i32); // 6
    let ni_sp = si.nctr * di; // 2
    let nj_sp = sj.nctr * dj; // 6

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");
    let common_orig = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    let rinv_orig = [env[PTR_RINV_ORIG], env[PTR_RINV_ORIG + 1], env[PTR_RINV_ORIG + 2]];
    let mut cintx = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend, "cg_sa10sa01", si.l, si.kappa, sj.l, sj.kappa, si.nprim, sj.nprim, si.nctr,
        sj.nctr, si.coord, sj.coord, common_orig, rinv_orig, &si.exps, &sj.exps, &si.coeff_row_major,
        &sj.coeff_row_major, &[], &[], &mut cintx,
    )
    .expect("cintx cg_sa10sa01 nctr1");

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    assert_eq!(vendor_CINTcgto_spinor(0, &bas) as usize, ni_sp);
    assert_eq!(vendor_CINTcgto_spinor(1, &bas) as usize, nj_sp);
    let shls = [0_i32, 1];
    let mut vendor = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    vendor_int1e_cg_sa10sa01_spinor(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

    let group_out = ni_sp * nj_sp * 2;
    let z = 1e-12;
    let mut total_mismatch = 0;
    eprintln!("\n=== sa01 SPINOR nctr=1 (ni_sp={ni_sp} nj_sp={nj_sp}) ===");
    for grp in 0..9 {
        let (mut both, mut half) = (0, 0);
        let mut mism = 0;
        for k in 0..group_out / 2 {
            let idx = grp * group_out + k * 2;
            let rz = cintx[idx].abs() <= z;
            let iz = cintx[idx + 1].abs() <= z;
            if !rz && !iz {
                both += 1;
            } else if rz != iz {
                half += 1;
            }
            if (cintx[idx] - vendor[idx]).abs() > 1e-9 || (cintx[idx + 1] - vendor[idx + 1]).abs() > 1e-9
            {
                mism += 1;
            }
        }
        total_mismatch += mism;
        eprintln!("  group {grp}: cintx both={both} half-zero={half}  mismatch-vs-vendor={mism}/{}", group_out / 2);
    }
    eprintln!("  TOTAL mismatch = {total_mismatch} / {}", ni_sp * nj_sp * 9);
    // Print first few elements of group 0 for the relationship.
    for k in 0..ni_sp * nj_sp {
        let idx = k * 2;
        eprintln!(
            "  g0 e{k}: cintx=({:+.5e},{:+.5e}) vendor=({:+.5e},{:+.5e})",
            cintx[idx], cintx[idx + 1], vendor[idx], vendor[idx + 1]
        );
    }
}

/// CONFIRMATION: the bug is the dispatcher hardcoding rinv=[0,0,0]. Call the launcher
/// DIRECTLY on the real fixture with rinv = env[PTR_RINV_ORIG] (the value vendor uses).
/// If this matches vendor, the fix is to plumb PTR_RINV_ORIG into the sa01 dispatch arms.
#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_rinv_center_is_the_bug() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::launch_int1e_sa10sa01_spinor_pair;
    use cintx_oracle::vendor_ffi::vendor_int1e_cg_sa10sa01_spinor;
    use cintx_runtime::{BackendIntent, BackendKind};

    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let si = extract_shell(0, &atm, &bas, &env);
    let sj = extract_shell(1, &atm, &bas, &env);
    let di = spinor_len_kappa(si.l as i32, si.kappa as i32);
    let dj = spinor_len_kappa(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");
    let common_orig = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    let dri = [
        si.coord[0] - common_orig[0],
        si.coord[1] - common_orig[1],
        si.coord[2] - common_orig[2],
    ];
    // The CORRECT rinv center from env (what vendor uses), NOT the dispatcher's [0,0,0].
    let rinv = [env[PTR_RINV_ORIG], env[PTR_RINV_ORIG + 1], env[PTR_RINV_ORIG + 2]];
    eprintln!("\n=== rinv center from env = {rinv:?} (dispatcher hardcodes [0,0,0]) ===");

    let mut cintx = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    launch_int1e_sa10sa01_spinor_pair::<f64>(
        &backend, si.l, si.kappa, sj.l, sj.kappa, si.nprim, sj.nprim, si.nctr, sj.nctr,
        si.coord, sj.coord, rinv, dri, &si.exps, &sj.exps, &si.coeff_row_major,
        &sj.coeff_row_major, &mut cintx,
    )
    .expect("direct launch with correct rinv");

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let shls = [0_i32, 1];
    let mut vendor = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    vendor_int1e_cg_sa10sa01_spinor(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

    let mism = count_mismatches(&cintx, &vendor, 1e-9, 0.0);
    eprintln!("  direct-launch(rinv=env) vs vendor: mismatch = {mism}/{}", ni_sp * nj_sp * 9 * 2);
    assert_eq!(
        mism, 0,
        "with rinv=env[PTR_RINV_ORIG] the sa01 spinor MUST match vendor — confirms the \
         dispatcher's hardcoded rinv=[0,0,0] is the bug"
    );
}

#[cfg(all(feature = "cpu", has_vendor_libcint))]
#[test]
fn sa01_spinor_pattern_probe() {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_oracle::vendor_ffi::{vendor_CINTcgto_spinor, vendor_int1e_cg_sa10sa01_spinor};
    use cintx_runtime::{BackendIntent, BackendKind};

    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let si = extract_shell(0, &atm, &bas, &env);
    let sj = extract_shell(1, &atm, &bas, &env);
    let di = spinor_len_kappa(si.l as i32, si.kappa as i32);
    let dj = spinor_len_kappa(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend");
    let common_orig = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    let rinv_orig = [env[PTR_RINV_ORIG], env[PTR_RINV_ORIG + 1], env[PTR_RINV_ORIG + 2]];
    let mut cintx = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend, "cg_sa10sa01", si.l, si.kappa, sj.l, sj.kappa, si.nprim, sj.nprim, si.nctr,
        sj.nctr, si.coord, sj.coord, common_orig, rinv_orig, &si.exps, &sj.exps, &si.coeff_row_major,
        &sj.coeff_row_major, &[], &[], &mut cintx,
    )
    .expect("cintx cg_sa10sa01");

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let vni = vendor_CINTcgto_spinor(0, &bas) as usize;
    let vnj = vendor_CINTcgto_spinor(1, &bas) as usize;
    assert_eq!((vni, vnj), (ni_sp, nj_sp), "spinor dims agree with vendor");
    let shls = [0_i32, 1];
    let mut vendor = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    vendor_int1e_cg_sa10sa01_spinor(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

    eprintln!("\n=== sa01 spinor pattern (ni_sp={ni_sp} nj_sp={nj_sp}, 9 groups) ===");
    let group_out = ni_sp * nj_sp * 2;
    let z = 1e-12;
    // Tally which simple relationship cintx↔vendor follows, element-wise.
    let (mut eq, mut irot, mut nirot, mut conj, mut other, mut bothzero) = (0, 0, 0, 0, 0, 0);
    for grp in 0..9 {
        for j in 0..nj_sp {
            for i in 0..ni_sp {
                let idx = grp * group_out + (j * ni_sp + i) * 2;
                let (cr, ci) = (cintx[idx], cintx[idx + 1]);
                let (vr, vi) = (vendor[idx], vendor[idx + 1]);
                let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 + 1e-6 * b.abs();
                let tag = if cr.abs() <= z && ci.abs() <= z && vr.abs() <= z && vi.abs() <= z {
                    bothzero += 1;
                    continue;
                } else if close(cr, vr) && close(ci, vi) {
                    eq += 1;
                    "eq"
                } else if close(cr, -vi) && close(ci, vr) {
                    irot += 1; // cintx = i * vendor
                    "i*vendor"
                } else if close(cr, vi) && close(ci, -vr) {
                    nirot += 1; // cintx = -i * vendor
                    "-i*vendor"
                } else if close(cr, vr) && close(ci, -vi) {
                    conj += 1;
                    "conj"
                } else {
                    other += 1;
                    "OTHER"
                };
                if grp < 2 || tag == "OTHER" {
                    eprintln!(
                        "  g{grp} j{j} i{i}: cintx=({cr:+.5e},{ci:+.5e}) vendor=({vr:+.5e},{vi:+.5e}) [{tag}]"
                    );
                }
            }
        }
    }
    eprintln!(
        "\n  TALLY: eq={eq}  i*vendor={irot}  -i*vendor={nirot}  conj={conj}  OTHER={other}  bothzero={bothzero}"
    );

    // Per-group: how many cintx elements have BOTH parts nonzero vs exactly one zero.
    eprintln!("\n  --- per-group cintx zero structure ---");
    for grp in 0..9 {
        let (mut both, mut half) = (0, 0);
        for j in 0..nj_sp {
            for i in 0..ni_sp {
                let idx = grp * group_out + (j * ni_sp + i) * 2;
                let rz = cintx[idx].abs() <= z;
                let iz = cintx[idx + 1].abs() <= z;
                if !rz && !iz {
                    both += 1;
                } else if rz != iz {
                    half += 1;
                }
            }
        }
        eprintln!("  group {grp}: cintx both-nonzero={both}  one-part-zero={half}");
    }

    // Hypothesis A: is cintx's |value| at least equal to vendor's |value| (i.e. only
    // a phase/assignment scramble, magnitudes preserved)? Compare sorted-abs per group.
    eprintln!("\n  --- per-group magnitude-set match (cintx vs vendor |·|, sorted) ---");
    for grp in 0..9 {
        let mut cm: Vec<f64> = Vec::new();
        let mut vm: Vec<f64> = Vec::new();
        for k in 0..group_out {
            cm.push(cintx[grp * group_out + k].abs());
            vm.push(vendor[grp * group_out + k].abs());
        }
        cm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let maxdev = cm.iter().zip(vm.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "  group {grp}: sorted-|·| max dev = {maxdev:.3e}  (cintx sum|·|={:.4e} vendor sum|·|={:.4e})",
            cm.iter().sum::<f64>(),
            vm.iter().sum::<f64>()
        );
    }
}

/// Sanity: the fixture rides distinct bra/ket coords and a non-zero gauge origin.
#[cfg(feature = "cpu")]
#[test]
fn fixture_is_well_formed() {
    let fx = build_cart_fixture(0, 0);
    assert_ne!(fx.ri, fx.rj, "bra/ket must be at distinct coords");
    assert!(fx.common_orig.iter().any(|&c| c != 0.0), "gauge origin must be non-zero");
    assert_eq!(fx.env[PTR_RINV_ORIG], 0.0, "rinv center pinned to 0");
}
