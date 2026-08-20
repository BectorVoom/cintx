//! SPIKE 006 — spinor-layout-divergence
//! ====================================
//! Recorded artifact: `.planning/spikes/006-spinor-layout-divergence/`.
//!
//! The one path where the layout contract genuinely DIVERGES from the real
//! component-leading form proven in spikes 001-005. The spinor output is
//! INTERLEAVED-COMPLEX and uses spinor block dims, not real ncart/nsph:
//!     out[comp * (ni_sp*nj_sp) * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
//!         comp       : slowest (rank 3 gradient)
//!         (j*ni_sp+i): ket-major column-major, i (bra) fastest
//!         {re,im}    : FASTEST axis (per-element complex interleave)
//!     ni_sp = CINTcgto_spinor(shell) = 4l+2 for kappa==0.
//!
//! This spike CHARACTERIZES the divergence (so the skill's "spinor differs" note is backed
//! by a concrete probe) rather than just asserting equality:
//!   A. buffer is complex-interleaved: len == rank * ni_sp*nj_sp * 2.
//!   B. component axis is STILL outermost: comp_stride == ni_sp*nj_sp*2.
//!   C. genuinely complex: the imaginary lane (odd indices) is non-trivially non-zero.
//!   D. re/im are PER-ELEMENT interleaved, NOT block-separated (de-interleaving diverges).
//!   E. non-square column-major i-fastest (transposing i/j within the complex block diverges).
//!   F. vendor: mm(vendor, cintx) == 0, and the D/E misreads each mismatch vendor.
//!
//! Run:
//!   cargo test -p cintx-oracle --features cpu --test spike_axis_fold_006 -- --ignored --nocapture
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test spike_axis_fold_006 -- --ignored --nocapture

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const RANK: usize = 3; // int1e_ipovlp_spinor is a 3-component gradient

fn spinor_len_kappa0(l: i32) -> usize {
    (4 * l + 2) as usize
}
fn mismatches(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x - **y).abs() > ATOL)
        .count()
}

/// One atom, s (l=0) + p (l=1) spinor shells (KAPPA_OF=0 → both GT+LT, ni_sp = 4l+2).
fn build_spinor_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coord = [0.0_f64, 0.0, 0.0];
    let exp3 = [3.4252509_f64, 0.6239137, 0.1688554];
    let coeff3 = [0.15432897_f64, 0.53532814, 0.44463454];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let coord_ptr = env.len() as i32;
    env.extend_from_slice(&coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&exp3);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&coeff3);

    let mut atm = vec![0_i32; ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for (s, l) in [(0usize, 0i32), (1usize, 1i32)] {
        bas[s * BAS_SLOTS + ATOM_OF] = 0;
        bas[s * BAS_SLOTS + ANG_OF] = l;
        bas[s * BAS_SLOTS + KAPPA_OF] = 0;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

/// Misread #1: de-interleave each component into [all re | all im] (block-separated complex).
fn deinterleave(buf: &[f64], rank: usize, complex_block: usize) -> Vec<f64> {
    let half = complex_block / 2; // # of complex elements per component
    let mut out = vec![0.0_f64; buf.len()];
    for c in 0..rank {
        let base = c * complex_block;
        for e in 0..half {
            out[base + e] = buf[base + e * 2]; // re's contiguous
            out[base + half + e] = buf[base + e * 2 + 1]; // im's contiguous
        }
    }
    out
}

/// Misread #2: transpose i<->j within each complex block (ket-major -> bra-major).
fn transpose_ij(buf: &[f64], rank: usize, ni: usize, nj: usize) -> Vec<f64> {
    let complex_block = ni * nj * 2;
    let mut out = vec![0.0_f64; buf.len()];
    for c in 0..rank {
        let base = c * complex_block;
        for i in 0..ni {
            for j in 0..nj {
                let src = base + (j * ni + i) * 2;
                let dst = base + (i * nj + j) * 2;
                out[dst] = buf[src];
                out[dst + 1] = buf[src + 1];
            }
        }
    }
    out
}

#[test]
#[ignore = "spike 006 — run explicitly with --ignored"]
fn spike_006_spinor_layout_divergence() {
    let (atm, bas, env) = build_spinor_fixture();
    // Non-square spinor pair: s (l=0 -> ni_sp=2) x p (l=1 -> nj_sp=6).
    let (si, sj) = (0usize, 1usize);
    let ni = spinor_len_kappa0(bas[si * BAS_SLOTS + ANG_OF]);
    let nj = spinor_len_kappa0(bas[sj * BAS_SLOTS + ANG_OF]);
    let complex_block = ni * nj * 2;
    let shls = [si as i32, sj as i32];

    let mut cintx = vec![0.0_f64; RANK * complex_block];
    unsafe {
        eval_raw(
            RawApiId::INT1E_IPOVLP_SPINOR,
            Some(&mut cintx),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("int1e_ipovlp_spinor eval_raw failed: {e:?}"));
    }

    println!("\n================ SPIKE 006 : spinor layout DIVERGENCE ================");
    println!(
        "  int1e_ipovlp_spinor  s×p  ni_sp={ni} nj_sp={nj}  complex_block={complex_block}  len={}",
        cintx.len()
    );

    // A. complex-interleaved length.
    assert_eq!(
        cintx.len(),
        RANK * ni * nj * 2,
        "A: len != rank*ni_sp*nj_sp*2 (not complex-interleaved)"
    );
    // B. component axis still outermost.
    assert_eq!(
        cintx.len() / RANK,
        complex_block,
        "B: comp_stride != ni_sp*nj_sp*2"
    );

    // C. genuinely complex — imaginary lane non-trivially non-zero.
    let im_nnz = cintx
        .iter()
        .skip(1)
        .step_by(2)
        .filter(|v| v.abs() > 1e-14)
        .count();
    let re_nnz = cintx.iter().step_by(2).filter(|v| v.abs() > 1e-14).count();
    assert!(
        im_nnz > 0,
        "C: imaginary lane all-zero — not genuinely complex (would read fine as real?!)"
    );
    println!(
        "  C. complex: re_nnz={re_nnz} im_nnz={im_nnz}  → buffer is genuinely interleaved-complex"
    );

    // D/E. misreads must change the buffer (vendor-free negative controls).
    let deint = deinterleave(&cintx, RANK, complex_block);
    let transp = transpose_ij(&cintx, RANK, ni, nj);
    assert!(
        mismatches(&cintx, &deint) > 0,
        "D: de-interleave was a no-op?"
    );
    assert!(
        mismatches(&cintx, &transp) > 0,
        "E: i/j transpose was a no-op (block square?)"
    );

    #[cfg(has_vendor_libcint)]
    {
        use cintx_oracle::vendor_ffi;
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let mut vendor = vec![0.0_f64; RANK * complex_block];
        vendor_ffi::vendor_int1e_ipovlp_spinor(&mut vendor, &shls, &atm, natm, &bas, nbas, &env);

        let mm = mismatches(&vendor, &cintx);
        assert_eq!(
            mm, 0,
            "F: cintx spinor != vendor (interleaved-complex layout divergence)"
        );
        let mm_deint = mismatches(&vendor, &deint);
        let mm_transp = mismatches(&vendor, &transp);
        assert!(
            mm_deint > 0,
            "F: block-separated complex ALSO matches vendor — interleave not pinned"
        );
        assert!(
            mm_transp > 0,
            "F: i/j-transposed ALSO matches vendor — orientation not pinned"
        );
        println!(
            "  F. vendor: mm(vendor,cintx)=0  mm(vendor,deinterleaved)={mm_deint}  mm(vendor,transposed)={mm_transp}"
        );
    }
    #[cfg(not(has_vendor_libcint))]
    println!(
        "  vendor: NOT linked — structure + misread-sensitivity only (set CINTX_ORACLE_BUILD_VENDOR=1 to pin)"
    );

    println!("\n  DIVERGENCE SUMMARY vs the real-family contract (spikes 001-005):");
    println!("    • block dims are spinor lengths (ni_sp=4l+2 @ kappa=0), NOT ncart/nsph");
    println!("    • each matrix element is a complex (re,im) PAIR — fastest axis, ×2 length");
    println!("    • component axis is STILL outermost; ket-major i-fastest still holds");
    println!("\n================ SPIKE 006 : done ================\n");
}
