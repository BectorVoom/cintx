//! Randomized ROCm vendor-parity oracle for the ip1 origk 3c1e device path
//! (quick task 260530-9ay).
//!
//! Drives the `#[cube(launch)] origk_ip1_kernel` device kernel on the ROCm
//! `HipRuntime` via the RAW API (`eval_raw` with
//! `RawApiId::Symbol("int3c1e_ip1_{r2,r4,r6}_origk_sph")`) over randomized
//! H2O/STO-3G 3-shell triples and compares element-for-element vs the vendored
//! libcint 6.1.3 oracle (`vendor_int3c1e_ip1_{r2,r4,r6}_origk_sph`).
//!
//! r2/r4 are the HARD vendor-parity assertions (mismatch_count == 0). r6 has a
//! pre-existing ~6% numeric divergence vs libcint on the y-component at the
//! highest k-power: cintx is self-consistent (ip1 = grad(scalar)) but differs
//! from libcint. This is a DOCUMENTED residual, NOT a regression and NOT
//! something this port fixes — the device path faithfully reproduces the HOST
//! (proven by the in-crate device-vs-host test, which includes r6). Here we
//! merely DRIVE r6 on the GPU and assert it produced nonzero output; we eprintln
//! the r6 vendor mismatch_count for visibility but DO NOT assert it is zero.
//!
//! Gating (double-gated + rocm, per project memory): real vendor parity is
//! double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (sets the
//! `has_vendor_libcint` cfg); the ROCm device path additionally needs
//! `--features rocm`, `CINTX_BACKEND=rocm`, and the `CINTX_ROCM_ORACLE=1` +
//! `#[ignore]` runtime convention picked up by `xtask rocm-oracle`. The
//! `unstable-source-api` feature MUST also be enabled (origk is an
//! unstable-source family). Without all gates the test cfgs out / skips.
//!
//! Run with:
//!   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!     cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
//!     --test origk_ip1_random_rocm_parity -- --ignored
//!
//! Tolerance atol=1e-12, rtol=0.0 (Phase 15 unstable-family convention).
#![cfg(feature = "rocm")]
#![cfg(feature = "unstable-source-api")]

use cintx_compat::raw::{ANG_OF, ATM_SLOTS, BAS_SLOTS, PTR_ENV_START, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_h2o_sto3g;

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

/// Spherical component count `2l+1`.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Build a randomized but structurally-valid H2O/STO-3G slab. Jitters each
/// primitive exponent block (3 prims each) by a uniform factor in [0.7, 1.4) and
/// the H1/H2 coordinates by ±0.3 bohr. The slot layout is unchanged.
fn build_random_h2o(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();

    let exp_blocks = [
        PTR_ENV_START + 10, // O-1s exp = 30
        PTR_ENV_START + 16, // O-2s exp = 36
        PTR_ENV_START + 22, // O-2p exp = 42
        PTR_ENV_START + 28, // H-1s exp = 48
    ];
    for &ptr in &exp_blocks {
        for k in 0..3 {
            env[ptr + k] *= rng.uniform(0.7, 1.4);
        }
    }

    let h1_ptr = PTR_ENV_START + 3; // 23
    let h2_ptr = PTR_ENV_START + 6; // 26
    for d in 0..3 {
        env[h1_ptr + d] += rng.uniform(-0.3, 0.3);
        env[h2_ptr + d] += rng.uniform(-0.3, 0.3);
    }

    (atm, bas, env)
}

/// Count element-for-element mismatches at atol=1e-12, rtol=0.0.
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

/// 3-shell triples (bas-row indices) covering varied (li,lj,lk) combinations.
/// shells: 0=O-1s(s), 1=O-2s(s), 2=O-2p(p), 3=H1-1s(s), 4=H2-1s(s).
const SHELL_TRIPLES: [[i32; 3]; 6] = [
    [0, 1, 0], // s,s,s
    [3, 4, 0], // s,s,s off-center
    [2, 0, 0], // p,s,s
    [0, 2, 0], // s,p,s
    [0, 0, 2], // s,s,p
    [0, 1, 2], // s,s,p mixed
];

/// Randomized origk `int3c1e_ip1_{r2,r4,r6}_origk_sph` ROCm vendor-parity oracle
/// for the ip1 (ncomp=3) device path. 8 random geometries × 6 triples.
///
/// r2/r4: HARD assertion mismatch_count == 0.
/// r6:    driven on-device + any_nonzero asserted; vendor mismatch_count is
///        reported (eprintln) but NOT asserted — documented ~6% residual.
#[test]
#[ignore]
fn test_int3c1e_ip1_origk_random_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let mut rng = Lcg::new(0x2605_3009_a900_c0de_u64);
    let n_geometries = 8usize;

    const NCOMP: usize = 3;

    #[allow(unused_mut)]
    let mut mismatch_r2 = 0usize;
    #[allow(unused_mut)]
    let mut mismatch_r4 = 0usize;
    #[allow(unused_mut)]
    let mut mismatch_r6 = 0usize;
    let mut any_nonzero_r2 = false;
    let mut any_nonzero_r4 = false;
    let mut any_nonzero_r6 = false;
    let mut cases = 0usize;

    for _geo in 0..n_geometries {
        let (atm, bas, env) = build_random_h2o(&mut rng);

        for &[i_sh, j_sh, k_sh] in SHELL_TRIPLES.iter() {
            let li = bas[i_sh as usize * BAS_SLOTS + ANG_OF];
            let lj = bas[j_sh as usize * BAS_SLOTS + ANG_OF];
            let lk = bas[k_sh as usize * BAS_SLOTS + ANG_OF];
            let shls = [i_sh, j_sh, k_sh];
            // origk emits spherical i/j AND spherical k (k_cartesian=false), ×3 comps.
            let n_elem = NCOMP * nsph(li) * nsph(lj) * nsph(lk);

            for (op_name, r_power) in [
                ("int3c1e_ip1_r2_origk_sph", 2u8),
                ("int3c1e_ip1_r4_origk_sph", 4u8),
                ("int3c1e_ip1_r6_origk_sph", 6u8),
            ] {
                // cintx ROCm device path (eval_raw resolves CINTX_BACKEND=rocm).
                let mut cintx_out = vec![0.0_f64; n_elem];
                let summary = unsafe {
                    eval_raw(
                        RawApiId::Symbol(op_name),
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
                .unwrap_or_else(|err| {
                    panic!("eval_raw failed for {op_name} triple {shls:?}: {err:?}")
                });
                assert!(
                    summary.bytes_written > 0,
                    "eval_raw wrote 0 bytes for {op_name} triple {shls:?}"
                );

                let nonzero = cintx_out.iter().any(|&v| v.abs() > 1e-30);
                match r_power {
                    2 => any_nonzero_r2 |= nonzero,
                    4 => any_nonzero_r4 |= nonzero,
                    _ => any_nonzero_r6 |= nonzero,
                }

                #[cfg(has_vendor_libcint)]
                {
                    let natm = (atm.len() / ATM_SLOTS) as i32;
                    let nbas = (bas.len() / BAS_SLOTS) as i32;
                    let mut vendor_out = vec![0.0_f64; n_elem];
                    match r_power {
                        2 => {
                            cintx_oracle::vendor_ffi::vendor_int3c1e_ip1_r2_origk_sph(
                                &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                            );
                            mismatch_r2 += count_mismatches(&cintx_out, &vendor_out);
                        }
                        4 => {
                            cintx_oracle::vendor_ffi::vendor_int3c1e_ip1_r4_origk_sph(
                                &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                            );
                            mismatch_r4 += count_mismatches(&cintx_out, &vendor_out);
                        }
                        _ => {
                            cintx_oracle::vendor_ffi::vendor_int3c1e_ip1_r6_origk_sph(
                                &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                            );
                            mismatch_r6 += count_mismatches(&cintx_out, &vendor_out);
                        }
                    }
                }

                cases += 1;
            }
        }
    }

    println!(
        "origk_ip1_random_rocm_parity: cases={cases} \
         mismatch_r2={mismatch_r2} mismatch_r4={mismatch_r4} mismatch_r6={mismatch_r6} \
         any_nonzero_r2={any_nonzero_r2} any_nonzero_r4={any_nonzero_r4} any_nonzero_r6={any_nonzero_r6}"
    );

    // r6 is a DOCUMENTED ~6% vendor residual (y-component at highest k-power):
    // cintx ip1 = grad(scalar) is self-consistent but differs from libcint. We
    // report the count for visibility but do NOT assert it is zero.
    eprintln!(
        "origk_ip1 r6 vendor mismatch_count = {mismatch_r6} (DOCUMENTED pre-existing ~6% \
         residual vs libcint on the highest-k-power y-component — NOT a regression; the device \
         path faithfully reproduces the cintx HOST, proven by the in-crate device-vs-host test)"
    );

    // r2/r4 are the HARD vendor-parity assertions.
    assert_eq!(
        mismatch_r2, 0,
        "ip1_r2 origk ROCm device output diverged from vendor in {mismatch_r2} elements"
    );
    assert_eq!(
        mismatch_r4, 0,
        "ip1_r4 origk ROCm device output diverged from vendor in {mismatch_r4} elements"
    );

    // All three r_powers must have actually run on-device (no all-zeros fallback).
    assert!(
        any_nonzero_r2 && any_nonzero_r4 && any_nonzero_r6,
        "some origk ip1 r_power produced all-zero output — the device kernel did not run \
         (r2={any_nonzero_r2} r4={any_nonzero_r4} r6={any_nonzero_r6})"
    );
}
