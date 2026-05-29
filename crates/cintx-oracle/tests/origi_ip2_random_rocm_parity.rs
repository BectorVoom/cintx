//! Randomized ROCm vendor-parity oracle for `int1e_{r2,r4}_origi_ip2_sph`
//! (quick task 260530-9ay).
//!
//! Exercises the CubeCL `origi_ip2_kernel` GPU device path (the origi `ip2`
//! derivative operators `int1e_r2_origi_ip2_sph` / `int1e_r4_origi_ip2_sph`,
//! ported to a `#[cube(launch)]` device kernel and dispatched on the ROCm
//! `HipRuntime` in quick-260530-9ay) under `CINTX_BACKEND=rocm` over many
//! RANDOMIZED H2O/STO-3G systems.
//!
//! The ip2 operators emit a 3-component (comp-slowest) output. For each random
//! system and each of the two operators, a varied shell-pair table is evaluated
//! on the ROCm device via the RAW API (`eval_raw` with
//! `RawApiId::Symbol("int1e_r2_origi_ip2_sph")` / `("int1e_r4_origi_ip2_sph")`
//! — the unstable-source families are driven through the raw API by Symbol, NOT
//! a raw-API enum variant) and compared ELEMENT-WISE against the vendored
//! libcint 6.1.3 oracle (`vendor_ffi::vendor_int1e_r2_origi_ip2_sph` /
//! `_r4_origi_ip2_sph`). The suite asserts `mismatch_count == 0` at
//! atol=1e-12 / rtol=0 AND `any_nonzero == true` (proving the device kernel
//! actually ran on the GPU, not an all-zeros fallback).
//!
//! Each case keeps the H2O/STO-3G slab structurally valid (same nbas/natm/slot
//! layout) but jitters the primitive exponents (scaled by Lcg::uniform(0.7, 1.4))
//! and the H atom coordinates (+/-0.3 bohr), producing a distinct valid system.
//!
//! Gated `#![cfg(feature = "rocm")]` + `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`
//! runtime gate. The vendor comparison is `#[cfg(has_vendor_libcint)]`. Trigger via:
//!   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!   cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
//!   --test origi_ip2_random_rocm_parity -- --ignored

#![cfg(feature = "rocm")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, BAS_SLOTS, NPRIM_OF, PTR_COORD, PTR_EXP, RawApiId, eval_raw,
};
use cintx_oracle::fixtures::build_h2o_sto3g;

/// Number of shells in the H2O STO-3G basis.
const N_SHELLS: usize = 5;

/// Number of output components for the ip2 operators.
const NCOMP: usize = 3;

/// AO count for a spherical shell with angular momentum l: 2l+1.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// LCG (Numerical Recipes constants) — reproducible without an external rng crate.
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

/// Build a fresh, randomized but structurally-valid H2O/STO-3G slab.
fn build_random_h2o(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();

    // Unique exponent blocks (ptr, nprim) deduped across shells.
    let mut exp_blocks: Vec<(usize, usize)> = Vec::new();
    for s in 0..N_SHELLS {
        let ptr = bas[s * BAS_SLOTS + PTR_EXP] as usize;
        let nprim = bas[s * BAS_SLOTS + NPRIM_OF] as usize;
        if !exp_blocks.iter().any(|&(p, _)| p == ptr) {
            exp_blocks.push((ptr, nprim));
        }
    }
    for &(ptr, nprim) in &exp_blocks {
        for k in 0..nprim {
            env[ptr + k] *= rng.uniform(0.7, 1.4);
        }
    }

    // Jitter the two H atom coordinates by +/-0.3 bohr (keep O at origin).
    for atom in 1..=2usize {
        let cptr = atm[atom * ATM_SLOTS + PTR_COORD] as usize;
        for d in 0..3 {
            env[cptr + d] += rng.uniform(-0.3, 0.3);
        }
    }

    (atm, bas, env)
}

/// Evaluate one origi ip2 shell-pair block (3-component, comp-slowest) via
/// `eval_raw` on the resolved (ROCm) backend (CINTX_BACKEND=rocm).
fn eval_origi_ip2_block(
    symbol: &'static str,
    shls: &[i32; 2],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let ni = nsph(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nsph(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let mut out = vec![0.0_f64; ni * nj * NCOMP];
    // SAFETY: atm/bas/env are well-formed by construction; shls in range.
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw {symbol} failed for shls {shls:?}: {e:?}"));
    }
    out
}

/// Randomized `int1e_{r2,r4}_origi_ip2_sph` ROCm vendor-parity oracle.
#[test]
#[ignore]
fn test_int1e_origi_ip2_random_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let n_systems = 12usize;
    let mut rng = Lcg::new(0x0719_2605_30c0_de9a_u64);

    let operators: [&str; 2] = ["int1e_r2_origi_ip2_sph", "int1e_r4_origi_ip2_sph"];

    // Varied shell-pair table covering s-s, s-p, p-s, p-(H s).
    let shell_pairs: [[i32; 2]; 4] = [[0, 1], [0, 2], [2, 0], [2, 3]];

    let mut case_count = 0usize;
    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for sys in 0..n_systems {
        let (atm, bas, env) = build_random_h2o(&mut rng);

        for &symbol in &operators {
            for shls in shell_pairs {
                case_count += 1;
                let cintx_out = eval_origi_ip2_block(symbol, &shls, &atm, &bas, &env);

                if cintx_out.iter().any(|&v| v.abs() > 1e-12) {
                    any_nonzero = true;
                }

                #[cfg(has_vendor_libcint)]
                {
                    use cintx_oracle::vendor_ffi;
                    let natm = (atm.len() / ATM_SLOTS) as i32;
                    let nbas = (bas.len() / BAS_SLOTS) as i32;
                    let mut vendor_out = vec![0.0_f64; cintx_out.len()];
                    match symbol {
                        "int1e_r2_origi_ip2_sph" => vendor_ffi::vendor_int1e_r2_origi_ip2_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            natm,
                            &bas,
                            nbas,
                            &env,
                        ),
                        "int1e_r4_origi_ip2_sph" => vendor_ffi::vendor_int1e_r4_origi_ip2_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            natm,
                            &bas,
                            nbas,
                            &env,
                        ),
                        _ => unreachable!(),
                    };
                    for (idx, (&r, &o)) in vendor_out.iter().zip(cintx_out.iter()).enumerate() {
                        let diff = (o - r).abs();
                        if diff > atol {
                            mismatch_count += 1;
                            eprintln!(
                                "  rocm origi-ip2 RANDOM MISMATCH sys {sys} op {symbol} shls {shls:?} \
                                 idx {idx}: vendor={r:.15e}, cintx={o:.15e}, diff={diff:.3e}, \
                                 atol={atol:.1e}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "rocm origi-ip2 oracle parity failed: {mismatch_count} mismatches across \
         {case_count} random int1e_{{r2,r4}}_origi_ip2_sph cases vs vendor at atol={atol:.0e}"
    );
    assert!(
        any_nonzero,
        "origi-ip2 device output was all zeros across {case_count} cases — \
         the device kernel did not run on the GPU"
    );
    println!(
        "  PASS: rocm int1e_{{r2,r4}}_origi_ip2_sph random vendor-parity mismatch_count=0 \
         across {case_count} cases (any_nonzero=true) at atol={atol:.0e}/rtol=0"
    );
}
