//! SPIKE 004 — multi-index-block-ordering
//! ======================================
//! Recorded artifact: `.planning/spikes/004-multi-index-block-ordering/`.
//!
//! Spikes 001/003 proved the component-leading offset formula
//!     out[comp * (ni*nj) + (j*ni + i)]
//! but ONLY for a 2-index (bra×ket) inner block. The skill states the contract as
//! universal. This spike exercises the generalization to 3- and 4-index families:
//!     2e   (4 indices): out[comp * (ni*nj*nk*nl) + (((l*nk+k)*nj+j)*ni+i)]  — i fastest
//!     3c2e (3 indices): out[comp * (ni*nj*nk)    + ((k*nj+j)*ni+i)]         — i fastest
//!
//! Method (generalizes the spike-003 i/j transpose-disagreement to N axes):
//!   Vendor-FREE: component-leading split (comp_stride == inner block; clean `rank`
//!     slices; no truncation) + order-sensitivity negative control
//!     (reindexing the inner block under any non-identity axis permutation CHANGES it).
//!   Vendor: mm(vendor, cintx) == 0  AND  for every non-identity permutation P,
//!     mm(vendor, reindex(cintx, P)) > 0 — so the documented i-fastest order is the
//!     one libcint uses, not a coincidence of a symmetric block.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_004 -- --ignored --nocapture
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test spike_axis_fold_004 -- --ignored --nocapture

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}
fn ang(bas: &[i32], s: usize) -> i32 {
    bas[s * BAS_SLOTS + ANG_OF]
}

fn mismatches(a: &[f64], b: &[f64]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| (**x - **y).abs() > ATOL).count()
}

/// s/p/d shells on two atoms — shells (atom,l): 0=(0,s) 1=(0,p) 2=(0,d) 3=(1,s) 4=(1,p) 5=(1,d).
fn build_spd_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let h0 = [0.0_f64, 0.0, -0.70];
    let h1 = [0.0_f64, 0.0, 0.70];
    let exp3 = [3.4252509_f64, 0.6239137, 0.1688554];
    let coeff3 = [0.15432897_f64, 0.53532814, 0.44463454];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let h0_ptr = env.len() as i32;
    env.extend_from_slice(&h0);
    let h1_ptr = env.len() as i32;
    env.extend_from_slice(&h1);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&exp3);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&coeff3);
    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 1;
    atm[PTR_COORD] = h0_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    let spec: [(i32, i32); 6] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    let mut bas = vec![0_i32; spec.len() * BAS_SLOTS];
    for (s, &(atom, l)) in spec.iter().enumerate() {
        bas[s * BAS_SLOTS + ATOM_OF] = atom;
        bas[s * BAS_SLOTS + ANG_OF] = l;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

/// Reindex one component-leading buffer's inner block under axis permutation `perm`.
///
/// `extents` are the per-axis sizes in claimed FASTEST→SLOWEST order (axis 0 = fastest).
/// The claimed flat inner index is `Σ idx[a] * Π_{b<a} extents[b]`.
/// `perm` is a permutation of `0..naxes`: the rebuilt buffer treats axis `perm[k]` as the
/// k-th fastest. `perm == identity` reproduces the input. Returns a buffer of equal length
/// where `out[claimed_idx] = buf[permuted_idx]`.
fn reindex(buf: &[f64], rank: usize, extents: &[usize], perm: &[usize]) -> Vec<f64> {
    let block: usize = extents.iter().product();
    let naxes = extents.len();
    // strides for claimed order (axis 0 fastest)
    let mut stride = vec![1usize; naxes];
    for a in 1..naxes {
        stride[a] = stride[a - 1] * extents[a - 1];
    }
    // strides for permuted order: axis perm[0] fastest
    let mut pstride = vec![1usize; naxes];
    let mut acc = 1usize;
    for k in 0..naxes {
        pstride[perm[k]] = acc;
        acc *= extents[perm[k]];
    }
    let mut out = vec![0.0_f64; buf.len()];
    let mut idx = vec![0usize; naxes];
    for c in 0..rank {
        let base = c * block;
        // iterate all multi-indices
        for flat in 0..block {
            let mut rem = flat;
            for a in 0..naxes {
                idx[a] = (rem / stride[a]) % extents[a];
                let _ = &mut rem;
            }
            let claimed: usize = (0..naxes).map(|a| idx[a] * stride[a]).sum();
            let permuted: usize = (0..naxes).map(|a| idx[a] * pstride[a]).sum();
            out[base + claimed] = buf[base + permuted];
        }
    }
    out
}

const RANK: usize = 3; // all probed families are first-derivative gradients

#[test]
#[ignore = "spike 004 — run explicitly with --ignored"]
fn spike_004_multi_index_block_ordering() {
    let (atm, bas, env) = build_spd_fixture();
    #[cfg(has_vendor_libcint)]
    let (natm, nbas) = ((atm.len() / ATM_SLOTS) as i32, (bas.len() / BAS_SLOTS) as i32);

    println!("\n================ SPIKE 004 : multi-index inner-block ordering ================");
    #[cfg(has_vendor_libcint)]
    println!("vendor: LINKED — ordering pinned against libcint");
    #[cfg(not(has_vendor_libcint))]
    println!("vendor: NOT linked — order-sensitivity negative control only");

    for rep in ["cart", "sph"] {
        let nf: fn(i32) -> usize = if rep == "cart" { ncart } else { nsph };
        println!("\n  ---- {rep} ----");

        // ===== 4-index : int2e_ip1, quartet (0,p)(0,d)(1,p)(1,d) — all axes > 1 =====
        {
            let shls = [1_i32, 2, 4, 5];
            let ext = [
                nf(ang(&bas, 1)), // ni (axis 0, fastest)
                nf(ang(&bas, 2)), // nj
                nf(ang(&bas, 4)), // nk
                nf(ang(&bas, 5)), // nl (slowest within block)
            ];
            let block: usize = ext.iter().product();
            let api = if rep == "cart" { RawApiId::INT2E_IP1_CART } else { RawApiId::INT2E_IP1_SPH };
            let mut cintx = vec![0.0_f64; RANK * block];
            unsafe {
                eval_raw(api, Some(&mut cintx), None, &shls, &atm, &bas, &env, None, None).unwrap();
            }
            // component-leading split
            assert_eq!(cintx.len(), RANK * block, "2e: len != rank*ni*nj*nk*nl");
            assert_eq!(cintx.len() / RANK, block, "2e: comp_stride != inner block");
            // negative controls: non-identity permutations must change the buffer
            let perms: [[usize; 4]; 3] = [[1, 0, 2, 3], [3, 2, 1, 0], [0, 1, 3, 2]];
            let mut neg_ok = true;
            for p in &perms {
                neg_ok &= mismatches(&cintx, &reindex(&cintx, RANK, &ext, p)) > 0;
            }
            assert!(neg_ok, "2e: a non-identity axis permutation left the block unchanged (degenerate fixture)");

            #[cfg(has_vendor_libcint)]
            {
                use cintx_oracle::vendor_ffi as v;
                let vfn = if rep == "cart" { v::vendor_int2e_ip1_cart } else { v::vendor_int2e_ip1_sph };
                let mut vendor = vec![0.0_f64; RANK * block];
                vfn(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);
                let mm0 = mismatches(&vendor, &cintx);
                assert_eq!(mm0, 0, "2e {rep}: cintx != vendor (inner-block order divergence)");
                let mut worst_perm_mm = usize::MAX;
                for p in &perms {
                    let mmp = mismatches(&vendor, &reindex(&cintx, RANK, &ext, p));
                    assert!(mmp > 0, "2e {rep}: permutation {p:?} ALSO matches vendor — order not pinned");
                    worst_perm_mm = worst_perm_mm.min(mmp);
                }
                println!(
                    "    int2e_ip1   ext(i,j,k,l)={ext:?} block={block:>4}  mm(vendor,cintx)=0  min mm(vendor,perm)={worst_perm_mm}"
                );
            }
            #[cfg(not(has_vendor_libcint))]
            println!("    int2e_ip1   ext(i,j,k,l)={ext:?} block={block:>4}  order-sensitive ✓");
        }

        // ===== 3-index : int3c2e_ip2, triple (0,p)(0,d)(1,p) =====
        {
            let shls = [1_i32, 2, 4];
            let ext = [nf(ang(&bas, 1)), nf(ang(&bas, 2)), nf(ang(&bas, 4))];
            let block: usize = ext.iter().product();
            let api = if rep == "cart" { RawApiId::INT3C2E_IP2_CART } else { RawApiId::INT3C2E_IP2_SPH };
            let mut cintx = vec![0.0_f64; RANK * block];
            unsafe {
                eval_raw(api, Some(&mut cintx), None, &shls, &atm, &bas, &env, None, None).unwrap();
            }
            assert_eq!(cintx.len() / RANK, block, "3c2e: comp_stride != inner block");
            let perms: [[usize; 3]; 2] = [[1, 0, 2], [2, 1, 0]];
            for p in &perms {
                assert!(mismatches(&cintx, &reindex(&cintx, RANK, &ext, p)) > 0, "3c2e: perm {p:?} no-op");
            }
            #[cfg(has_vendor_libcint)]
            {
                use cintx_oracle::vendor_ffi as v;
                let vfn = if rep == "cart" { v::vendor_int3c2e_ip2_cart } else { v::vendor_int3c2e_ip2_sph };
                let mut vendor = vec![0.0_f64; RANK * block];
                vfn(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);
                assert_eq!(mismatches(&vendor, &cintx), 0, "3c2e {rep}: cintx != vendor");
                let mut worst = usize::MAX;
                for p in &perms {
                    let mmp = mismatches(&vendor, &reindex(&cintx, RANK, &ext, p));
                    assert!(mmp > 0, "3c2e {rep}: perm {p:?} also matches vendor — order not pinned");
                    worst = worst.min(mmp);
                }
                println!("    int3c2e_ip2 ext(i,j,k)={ext:?}   block={block:>4}  mm(vendor,cintx)=0  min mm(vendor,perm)={worst}");
            }
            #[cfg(not(has_vendor_libcint))]
            println!("    int3c2e_ip2 ext(i,j,k)={ext:?}   block={block:>4}  order-sensitive ✓");
        }
    }
    println!("\n================ SPIKE 004 : done ================\n");
}
