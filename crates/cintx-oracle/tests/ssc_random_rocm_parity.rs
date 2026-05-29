//! Randomized ROCm vendor-parity oracle for the SCALAR `int3c2e_sph_ssc`
//! (spin-spin contact 3-center 2e) device path on the ROCm backend
//! (quick task 260529-twi Task 6).
//!
//! Drives the `ssc_scalar_kernel` `#[cube(launch)]` device kernel on the ROCm
//! `HipRuntime` via the RAW API (`eval_raw` with `RawApiId::Symbol(
//! "int3c2e_sph_ssc")`) over randomized H2O/STO-3G 3-shell triples and compares
//! the output element-for-element vs the vendored libcint 6.1.3 oracle
//! (`vendor_int3c2e_sph_ssc`) — proving both "the device kernel ran on the GPU"
//! AND "0 divergence vs libcint".
//!
//! SSC c2s: i,j → spherical; k stays Cartesian. So the per-triple output element
//! count is `nsph(li) * nsph(lj) * ncart(lk)`.
//!
//! Gating (double-gated + rocm + unstable-source-api): real vendor parity is
//! double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (the latter
//! sets the `has_vendor_libcint` cfg); the ROCm device path additionally needs
//! `--features rocm,unstable-source-api`, `CINTX_BACKEND=rocm`, and the
//! `CINTX_ROCM_ORACLE=1` + `#[ignore]` runtime convention picked up by
//! `xtask rocm-oracle`. Without all gates the test cfgs out / skips.
//!
//! Run with:
//!   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!     cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
//!     --test ssc_random_rocm_parity -- --ignored
//!
//! Tolerance atol=1e-12, rtol=0.0 (Phase 15).
#![cfg(feature = "rocm")]
#![cfg(feature = "unstable-source-api")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

/// Deterministic LCG (Numerical Recipes constants) — reproducible random suite
/// without an external rng crate.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform f64 in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + frac * (hi - lo)
    }
}

const N_SHELLS: usize = 5;

/// Cartesian component count `(l+1)(l+2)/2`.
fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// Spherical component count `2l+1`.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// H2O STO-3G raw atm/bas/env builder (PTR_ENV_START-aligned; copied verbatim
/// from two_electron_random_rocm_parity.rs). bas rows: 0=O-1s(l=0), 1=O-2s(l=0),
/// 2=O-2p(l=1), 3=H1-1s(l=0), 4=H2-1s(l=0). Returns the exponent-block env
/// offsets so the per-case randomizer can jitter them in place.
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>, [usize; 4]) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START]; // zeros for reserved global slots

    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);

    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_coeff);
    let o2s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_coeff);
    let o2p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_coeff);
    let h1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; N_SHELLS * BAS_SLOTS];
    // Shell 0: O 1s.
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = o1s_exp_ptr;
    bas[PTR_COEFF] = o1s_coeff_ptr;
    // Shell 1: O 2s.
    bas[BAS_SLOTS + ATOM_OF] = 0;
    bas[BAS_SLOTS + ANG_OF] = 0;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;
    // Shell 2: O 2p (l=1).
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;
    // Shell 3: H1 1s.
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;
    // Shell 4: H2 1s (shares the H 1s exponent/coeff block).
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    let exp_blocks = [
        o1s_exp_ptr as usize,
        o2s_exp_ptr as usize,
        o2p_exp_ptr as usize,
        h1s_exp_ptr as usize,
    ];

    (atm, bas, env, exp_blocks)
}

/// Build a randomized but structurally-valid H2O/STO-3G slab. Jitters each
/// primitive exponent block (3 prims each) by a uniform factor in [0.7, 1.4) and
/// the H1/H2 coordinates by ±0.3 bohr. The slot layout is unchanged.
fn build_random_h2o(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env, exp_blocks) = build_h2o_sto3g();

    for &ptr in &exp_blocks {
        for k in 0..3 {
            env[ptr + k] *= rng.uniform(0.7, 1.4);
        }
    }

    // H1 coord block at PTR_ENV_START+3, H2 at PTR_ENV_START+6.
    let h1_ptr = PTR_ENV_START + 3;
    let h2_ptr = PTR_ENV_START + 6;
    for d in 0..3 {
        env[h1_ptr + d] += rng.uniform(-0.3, 0.3);
        env[h2_ptr + d] += rng.uniform(-0.3, 0.3);
    }

    (atm, bas, env)
}

/// Count element-for-element mismatches at atol=1e-12, rtol=0.0 (Phase 15).
#[cfg(has_vendor_libcint)]
fn count_mismatches(cintx: &[f64], vendor: &[f64]) -> usize {
    assert_eq!(
        cintx.len(),
        vendor.len(),
        "length mismatch: cintx={} vendor={}",
        cintx.len(),
        vendor.len()
    );
    cintx
        .iter()
        .zip(vendor.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-12)
        .count()
}

/// ssc 3-shell triple sweep (bas-row indices). All-s and s/p mixed triples
/// exercise the n-recurrence, the ij-HRR split (lj>0), and the ket m-recurrence
/// (lk>0). Triples are NOT pre-canonicalized — launch_ssc does the ibase swap +
/// transpose internally, so the un-swap host path is exercised too.
const SHELL_TRIPLES: [[i32; 3]; 6] = [
    [3, 4, 0], // H1-1s,H2-1s,O-1s — all s
    [0, 1, 2], // O-1s,O-2s,O-2p   — s/s/p (k Cartesian p)
    [0, 2, 1], // O-1s,O-2p,O-2s   — s/p/s (ij-HRR split via lj>0)
    [2, 0, 1], // O-2p,O-1s,O-2s   — p/s/s (i raise; no swap)
    [0, 2, 2], // O-1s,O-2p,O-2p   — s/p/p (split + ket recur)
    [2, 2, 0], // O-2p,O-2p,O-1s   — p/p/s (split with i+1 mixing)
];

/// Randomized `int3c2e_sph_ssc` ROCm vendor-parity oracle for the scalar ssc
/// device path. 48 cases (8 random geometries × 6 triples).
#[test]
#[ignore]
fn test_int3c2e_sph_ssc_random_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let mut rng = Lcg::new(0x2605_2917_2604_0529_u64);
    let n_geometries = 8usize;

    #[allow(unused_mut)]
    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut cases = 0usize;

    for _geo in 0..n_geometries {
        let (atm, bas, env) = build_random_h2o(&mut rng);

        for &[i_sh, j_sh, k_sh] in SHELL_TRIPLES.iter() {
            let li = bas[i_sh as usize * BAS_SLOTS + ANG_OF];
            let lj = bas[j_sh as usize * BAS_SLOTS + ANG_OF];
            let lk = bas[k_sh as usize * BAS_SLOTS + ANG_OF];
            let shls = [i_sh, j_sh, k_sh];

            // SSC: i,j spherical, k Cartesian.
            let n_elem = nsph(li) * nsph(lj) * ncart(lk);

            // cintx ROCm device path (eval_raw resolves CINTX_BACKEND=rocm).
            let mut cintx_out = vec![0.0_f64; n_elem];
            let summary = unsafe {
                eval_raw(
                    RawApiId::Symbol("int3c2e_sph_ssc"),
                    Some(&mut cintx_out),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
            }
            .unwrap_or_else(|err| panic!("eval_raw failed for triple {shls:?}: {err:?}"));
            assert!(
                summary.bytes_written > 0,
                "eval_raw wrote 0 bytes for {shls:?}"
            );

            if cintx_out.iter().any(|&v| v.abs() > 1e-30) {
                any_nonzero = true;
            }

            #[cfg(has_vendor_libcint)]
            {
                let natm = (atm.len() / ATM_SLOTS) as i32;
                let nbas = (bas.len() / BAS_SLOTS) as i32;
                let mut vendor_out = vec![0.0_f64; n_elem];
                cintx_oracle::vendor_ffi::vendor_int3c2e_sph_ssc(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );
                mismatch_count += count_mismatches(&cintx_out, &vendor_out);
            }

            cases += 1;
        }
    }

    println!(
        "ssc_random_rocm_parity: cases={cases} mismatch_count={mismatch_count} any_nonzero={any_nonzero}"
    );
    assert_eq!(
        mismatch_count, 0,
        "scalar int3c2e_sph_ssc ROCm device output diverged from vendor in {mismatch_count} elements across {cases} cases"
    );
    assert!(
        any_nonzero,
        "all int3c2e_sph_ssc outputs were zero — the device kernel did not actually run (all-zeros fallback)"
    );
}
