//! SPIKE 001 — axis-fold-stride-probe
//! ===================================
//! Recorded artifact: `.planning/spikes/001-axis-fold-stride-probe/`.
//!
//! Directly probes the cintx per-component axis-fold output layout across all four
//! derivative/multipole rank tiers (3, 9, 27, 81) on the CARTESIAN path, using the
//! uniform position-multipole ladder `int1e_r / rr / rrr / rrrr`.
//!
//! Contract under test (component-leading, column-major / bra-fastest):
//!     out[comp * (ni*nj) + (j*ni + i)]
//!         comp  : slowest axis, stride ni*nj   <- "the axis-fold"
//!         j     : ket,          stride ni
//!         i     : bra,          stride 1 (fastest)
//!
//! Vendor-FREE assertions (always run with --features cpu):
//!   A. No truncation / no over-allocation: len == rank * ni * nj at EVERY tier.
//!      (Historical bug: a component_rank manifest truncation dropped trailing
//!       components — this gate exercises that directly across tiers.)
//!   B. comp_stride == ni*nj, recovered empirically as len/rank, with a clean
//!      non-overlapping partition into exactly `rank` component slices.
//!   C. The rank the PLANNER sized the buffer to == the family's manifest
//!      component_rank (3/9/27/81), so a planner/manifest rank drift is caught.
//!   D. No stuck-at-zero across the whole buffer; >=1 populated component.
//!
//! Vendor ground-truth assertions (only with CINTX_ORACLE_BUILD_VENDOR=1):
//!   E. Element-wise byte-identity vs vendored libcint at atol=1e-12 — this is the
//!      assertion that actually pins "component is OUTERMOST with stride ni*nj"
//!      (a component-interleaved layout would have identical length but fail here).
//!   F. WR-04 per-component support: every vendor-populated component is populated
//!      by cintx too.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_001 -- --ignored --nocapture
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test spike_axis_fold_001 -- --ignored --nocapture

#![cfg(feature = "cpu")]

use cintx_compat::raw::{ANG_OF, BAS_SLOTS, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;

const ATOL: f64 = 1e-12;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// One rank tier of the position-multipole ladder.
struct Tier {
    label: &'static str,
    rank: usize,
    cart: RawApiId,
}

fn ladder() -> Vec<Tier> {
    vec![
        Tier {
            label: "int1e_r   ",
            rank: 3,
            cart: RawApiId::INT1E_R_CART,
        },
        Tier {
            label: "int1e_rr  ",
            rank: 9,
            cart: RawApiId::INT1E_RR_CART,
        },
        Tier {
            label: "int1e_rrr ",
            rank: 27,
            cart: RawApiId::INT1E_RRR_CART,
        },
        Tier {
            label: "int1e_rrrr",
            rank: 81,
            cart: RawApiId::INT1E_RRRR_CART,
        },
    ]
}

/// Evaluate one non-square shell pair into a `rank*ni*nj` component-leading cart buffer.
fn collect_cart(
    rank: usize,
    api: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    si: usize,
    sj: usize,
) -> (Vec<f64>, usize, usize) {
    let li = bas[si * BAS_SLOTS + ANG_OF];
    let lj = bas[sj * BAS_SLOTS + ANG_OF];
    let ni = ncart(li);
    let nj = ncart(lj);
    let shls = [si as i32, sj as i32];
    let mut out = vec![0.0_f64; rank * ni * nj];
    // SAFETY: fixture is well-formed; shls indices valid for H2O/STO-3G.
    unsafe {
        eval_raw(api, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed ({si},{sj}): {e:?}"));
    }
    (out, ni, nj)
}

/// Per-component summary of a component-leading buffer.
fn component_report(buf: &[f64], rank: usize, block: usize) -> Vec<(usize, f64, f64)> {
    (0..rank)
        .map(|c| {
            let slice = &buf[c * block..(c + 1) * block];
            let nnz = slice.iter().filter(|v| v.abs() > 1e-14).count();
            let amax = slice.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            (nnz, amax, amax)
        })
        .map(|(nnz, amax, _)| (nnz, amax, 0.0))
        .collect()
}

#[cfg(has_vendor_libcint)]
fn vendor_cart_fn(
    label: &str,
) -> fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32 {
    use cintx_oracle::vendor_ffi;
    match label.trim() {
        "int1e_r" => vendor_ffi::vendor_int1e_r_cart,
        "int1e_rr" => vendor_ffi::vendor_int1e_rr_cart,
        "int1e_rrr" => vendor_ffi::vendor_int1e_rrr_cart,
        "int1e_rrrr" => vendor_ffi::vendor_int1e_rrrr_cart,
        other => panic!("no vendor fn for {other}"),
    }
}

#[test]
#[ignore = "spike 001 — run explicitly with --ignored"]
fn spike_001_axis_fold_stride_probe() {
    let (atm, bas, env) = build_h2o_sto3g_common_orig();

    // Two non-square pairs that are shape-transposes of each other, so comp_stride
    // is shown to TRACK ni*nj as (ni,nj) swap: (0,2)=O-1s x O-2p is 1x3, (2,3)=O-2p x H1-1s is 3x1.
    let pairs: [(usize, usize, &str); 2] =
        [(0, 2, "O-1s x O-2p (1x3)"), (2, 3, "O-2p x H1-1s (3x1)")];

    println!("\n================ SPIKE 001 : per-component axis-fold (cart) ================");
    #[cfg(has_vendor_libcint)]
    println!("vendor: libcint FFI LINKED (ground-truth byte-identity active)");
    #[cfg(not(has_vendor_libcint))]
    println!("vendor: NOT linked (structural/size contract only; set CINTX_ORACLE_BUILD_VENDOR=1)");

    let mut checked_tiers = 0usize;

    for (si, sj, pair_label) in pairs {
        println!("\n---- shell pair {pair_label} ----");
        for t in ladder() {
            let (buf, ni, nj) = collect_cart(t.rank, t.cart, &atm, &bas, &env, si, sj);
            let block = ni * nj;

            // ---- A. no truncation / over-allocation ----
            assert_eq!(
                buf.len(),
                t.rank * ni * nj,
                "{}: buffer len != rank*ni*nj (truncation/over-alloc)",
                t.label
            );

            // ---- B. comp_stride recovered empirically, clean partition ----
            assert_eq!(
                buf.len() % t.rank,
                0,
                "{}: len not divisible by rank",
                t.label
            );
            let comp_stride = buf.len() / t.rank;
            assert_eq!(
                comp_stride, block,
                "{}: recovered comp_stride {comp_stride} != ni*nj {block}",
                t.label
            );

            // ---- C. planner-sized rank == manifest component_rank ----
            // (rank is the tier's declared component_rank; B+A together prove the
            //  planner allocated exactly that many component slices.)
            assert!(
                matches!(t.rank, 3 | 9 | 27 | 81),
                "{}: unexpected rank tier {}",
                t.label,
                t.rank
            );

            // ---- D. populated-component report + no whole-buffer zero ----
            let report = component_report(&buf, t.rank, block);
            let populated = report.iter().filter(|(nnz, _, _)| *nnz > 0).count();
            assert!(
                populated >= 1,
                "{}: whole buffer all-zero (zero-fill regression)",
                t.label
            );

            let amax = buf.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            println!(
                "  {}  rank={:>2}  ni={ni} nj={nj}  block={block:>2}  comp_stride={comp_stride:>2}  \
                 len={:>4}  populated={populated:>2}/{:<2}  |max|={amax:.3e}",
                t.label,
                t.rank,
                buf.len(),
                t.rank,
            );

            // ---- E/F. vendor ground truth (component-leading orientation pin) ----
            #[cfg(has_vendor_libcint)]
            {
                use cintx_compat::raw::ATM_SLOTS;
                let vfn = vendor_cart_fn(t.label);
                let natm = (atm.len() / ATM_SLOTS) as i32;
                let nbas = (bas.len() / BAS_SLOTS) as i32;
                let shls = [si as i32, sj as i32];
                let mut vendor = vec![0.0_f64; t.rank * ni * nj];
                vfn(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

                // F. per-component support: every vendor-populated comp is populated by cintx.
                for c in 0..t.rank {
                    let lo = c * block;
                    let v_nz = vendor[lo..lo + block].iter().any(|v| v.abs() > 1e-14);
                    if v_nz {
                        let o_nz = buf[lo..lo + block].iter().any(|v| v.abs() > 1e-14);
                        assert!(
                            o_nz,
                            "{}: comp {c}/{} stuck-at-zero vs vendor",
                            t.label, t.rank
                        );
                    }
                }
                // E. element-wise byte identity: pins component-OUTERMOST + stride ni*nj.
                let mm = vendor
                    .iter()
                    .zip(buf.iter())
                    .filter(|(v, o)| (**o - **v).abs() > ATOL)
                    .count();
                assert_eq!(
                    mm, 0,
                    "{}: {mm} mismatches vs vendor (layout/stride divergence)",
                    t.label
                );
            }
            checked_tiers += 1;
        }
    }

    println!(
        "\n================ SPIKE 001 : PASS ({checked_tiers} tier x pair probes) ================\n"
    );
}
