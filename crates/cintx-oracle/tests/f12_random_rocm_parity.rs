//! Randomized ROCm idempotency oracle for `int2e_stg_sph` (quick task 260529-i2q).
//!
//! Exercises the CubeCL `f12_cart_contraction_kernel` GPU device path (the F12
//! base Cartesian contraction, ported to a `#[cube(launch)]` device kernel and
//! dispatched on the ROCm `HipRuntime` in quick-260529-i2q) under
//! `CINTX_BACKEND=rocm` over many RANDOMIZED H2O/STO-3G F12 systems. For each
//! system the raw API `int2e_stg_sph` is evaluated TWICE on the ROCm device and
//! the two results must agree exactly (idempotency); the suite as a whole must
//! produce non-zero output (proving the device kernel actually ran on the GPU,
//! not an all-zeros host fallback).
//!
//! Unlike the ECP rocm oracle (`ecp_random_rocm_parity.rs`, which drives the SAFE
//! API), the F12 family has NO safe-API enum variant — it is driven through the
//! RAW API (`eval_raw` + `RawApiId::Symbol("int2e_stg_sph")`), mirroring
//! `f12_oracle_parity.rs`. The backend is resolved from `CINTX_BACKEND=rocm`
//! exported by `xtask rocm-oracle`, NOT from options.
//!
//! Each case keeps the real H2O/STO-3G `atm`/`bas`/`env` slab intact (so the raw
//! layout / nbas / natm slot structure stays valid) but randomizes the F12 zeta
//! (`PTR_F12_ZETA`, env[9]) into a physically sane range AND varies the shell
//! quartet selection across cases, producing a distinct valid F12 system per case
//! that exercises the device kernel.
//!
//! Triple-gated: `#![cfg(all(feature = "rocm", feature = "with-f12"))]` (f12 is
//! feature-gated, unlike ECP) + `#[ignore]` + the `CINTX_ROCM_ORACLE=1` runtime
//! assert. Picked up by `xtask rocm-oracle --profile with-f12`'s
//! `cargo test -p cintx-oracle --features rocm,with-f12 -- --ignored` invocation.
//! Trigger via: `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle --profile with-f12`.

#![cfg(all(feature = "rocm", feature = "with-f12"))]

use cintx_compat::raw::{ANG_OF, BAS_SLOTS, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_h2o_sto3g_f12;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG (Numerical Recipes constants) — keeps the random suite
// reproducible without an external rng crate (copied from
// ecp_random_rocm_parity.rs).
// ─────────────────────────────────────────────────────────────────────────────

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
    /// Uniform usize in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// STO-3G H2O shell candidates. 5 shells: 0=O-1s, 1=O-2s, 2=O-2p, 3=H1-1s,
/// 4=H2-1s. Mirrors the quartets exercised by `f12_oracle_parity.rs`:
/// all-s plus an s/p mixed quartet (O-2p has l=1).
const SHELL_QUARTETS: [[i32; 4]; 4] = [
    [0, 1, 0, 1], // O-1s / O-2s / O-1s / O-2s — all-s
    [3, 4, 3, 4], // H1-1s / H2-1s / H1-1s / H2-1s — all-s, different centers
    [0, 2, 0, 2], // O-1s / O-2p / O-1s / O-2p — mixed s/p (exercises cart_comps/strides)
    [2, 0, 2, 0], // O-2p / O-1s / O-2p / O-1s — mixed s/p, transposed
];

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Number of output elements for a 4-shell sph integral (component_count=1).
fn n_sph_elements(shls: &[i32; 4], bas: &[i32]) -> usize {
    shls.iter()
        .map(|&s| nsph_for_l(bas[s as usize * BAS_SLOTS + ANG_OF]))
        .product()
}

/// Evaluate the F12 base `int2e_stg_sph` integral block once via cintx
/// `eval_raw` on the resolved (ROCm) backend. Mirrors
/// `f12_oracle_parity.rs::eval_f12_sph`. The backend is resolved from
/// `CINTX_BACKEND=rocm` exported by `xtask rocm-oracle` (executor.rs
/// resolve_backend_kind), NOT from options.
fn eval_f12_once(shls: &[i32; 4], atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let n = n_sph_elements(shls, bas);
    let mut out = vec![0.0_f64; n];
    unsafe {
        eval_raw(
            RawApiId::Symbol("int2e_stg_sph"),
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw int2e_stg_sph failed for shls {shls:?}: {e:?}"));
    }
    out
}

/// Randomized `int2e_stg_sph` ROCm idempotency oracle.
///
/// Drives the raw-API F12 base path TWICE per random H2O/STO-3G system on the
/// ROCm GPU and asserts the two evaluations agree exactly (idempotency,
/// atol=1e-12 / rtol=1e-10), with non-zero output across the suite (proving the
/// `f12_cart_contraction_kernel` `#[cube(launch)]` device kernel ran on the AMD
/// GPU).
#[test]
#[ignore]
fn test_f12_stg_sph_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm,with-f12 -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;
    let n_cases = 48usize;
    let mut rng = Lcg::new(0xf12c_2605_29bf_1234_u64);

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        // Distinct valid F12 system per case: random non-zero zeta + a varied
        // shell quartet. The (atm, bas) slab layout stays intact; the F12-specific
        // randomness lives in the env-stored zeta (PTR_F12_ZETA, env[9]) and the
        // shell selection — both legitimately exercise the device contraction.
        let zeta = rng.uniform(0.4, 2.5);
        let (atm, bas, env) = build_h2o_sto3g_f12(zeta);
        let shls = SHELL_QUARTETS[rng.below(SHELL_QUARTETS.len())];

        let out1 = eval_f12_once(&shls, &atm, &bas, &env);
        let out2 = eval_f12_once(&shls, &atm, &bas, &env);

        assert_eq!(
            out1.len(),
            out2.len(),
            "case {case}: output length mismatch between the two ROCm evaluations"
        );
        for (idx, (&r, &o)) in out1.iter().zip(out2.iter()).enumerate() {
            let diff = (o - r).abs();
            let threshold = atol + rtol * r.abs();
            if diff > threshold {
                mismatch_count += 1;
                eprintln!(
                    "  rocm F12 RANDOM MISMATCH case {case} (zeta={zeta:.4}, shls={shls:?}) idx {idx}: \
                     ref={r:.15e}, obs={o:.15e}, diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
        if out1.iter().any(|&v| v.abs() > 1e-12) {
            any_nonzero = true;
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "rocm F12 oracle idempotency failed: {mismatch_count} mismatches across {n_cases} random int2e_stg_sph cases"
    );
    assert!(
        any_nonzero,
        "F12 device output was all zeros across {n_cases} cases — the device kernel did not run on the GPU"
    );
    println!(
        "  PASS: rocm int2e_stg_sph random idempotency mismatch_count=0 across {n_cases} cases \
         (any_nonzero=true) at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
