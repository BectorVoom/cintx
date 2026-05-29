//! Randomized ROCm idempotency oracle for `int1e_ecp_sph` (quick task 260529-gbf).
//!
//! Exercises the CubeCL `ecp_angular_kernel` GPU device path (the Type-1 angular
//! splice, ported to a `#[cube(launch)]` device kernel and dispatched on the
//! ROCm `HipRuntime` in quick-260529-gbf) under `CINTX_BACKEND=rocm` over many
//! RANDOMIZED Cu/LANL2DZ ECP systems. For each system the safe API
//! `OperatorId::INT1E_ECP_SPH` is evaluated TWICE on the ROCm device and the two
//! results must agree exactly (idempotency); the suite as a whole must produce
//! non-zero output (proving the device kernel actually ran on the GPU, not an
//! all-zeros host fallback).
//!
//! Unlike the `center_*_parity.rs` rocm oracles (which drive `eval_raw`), the ECP
//! family is driven through the SAFE API (`SessionRequest` + typed `BasisSet`),
//! mirroring `safe_api_ecp_parity.rs` — ECP has no raw-API enum variant.
//!
//! Each case keeps the real Cu/LANL2DZ `EcpShell` slab intact (so Type-1 Local +
//! Type-2 Projected channels both fire) but randomizes the AO shell exponents /
//! contraction coefficients and the atom coordinate within physically sane
//! ranges, producing a distinct valid ECP system per case.
//!
//! Gated `#[cfg(feature = "rocm")]` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`.
//! ECP runs under the BASE rocm profile (no with-* feature), so this file is
//! picked up by `xtask rocm-oracle`'s
//! `cargo test -p cintx-oracle --features rocm -- --ignored` invocation.
//! Trigger via: `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`.

#![cfg(feature = "rocm")]

use cintx_core::ecp::{EcpChannel, EcpShell};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG (Numerical Recipes constants) — keeps the random suite
// reproducible without an external rng crate (copied from center_4c1e_parity.rs).
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
}

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Cu/LANL2DZ ECP system builder.
//
// Reads the fixture JSON once (the same `data/cu_lanl2dz.json` that
// safe_api_ecp_parity.rs reads), then per case builds a fresh typed BasisSet:
//   - AO shell exponents jittered into [0.25, 4.0], coefficients into [0.15, 1.0]
//   - the atom coordinate jittered by ±0.5 bohr
//   - the EcpShell slab kept VERBATIM (radial powers / exponents / coefficients
//     unchanged) so the ECP itself stays a valid LANL2DZ potential and both the
//     Local (Type-1) and Projected (Type-2) channels fire.
// Returns the typed BasisSet plus the AO Shell vector for shell-pair iteration.
// ─────────────────────────────────────────────────────────────────────────────

fn build_random_ecp_system(rng: &mut Lcg) -> (BasisSet, Vec<Arc<Shell>>) {
    let raw = std::fs::read_to_string("data/cu_lanl2dz.json")
        .expect("Cu/LANL2DZ fixture JSON missing — required for the ROCm ECP oracle");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("Cu/LANL2DZ JSON invalid");

    let coord_arr = parsed["atom"]["coord"].as_array().unwrap();
    // Jitter the atom coordinate (keep it finite + away from pathological 0).
    let coord: [f64; 3] = [
        coord_arr[0].as_f64().unwrap() + rng.uniform(-0.5, 0.5),
        coord_arr[1].as_f64().unwrap() + rng.uniform(-0.5, 0.5),
        coord_arr[2].as_f64().unwrap() + rng.uniform(-0.5, 0.5),
    ];
    let z = parsed["atom"]["Z"].as_i64().unwrap() as u16;

    let atom = Atom::try_new(z, coord, NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom].into_boxed_slice());

    // AO shells — randomize exponents and contraction coefficients per primitive
    // (preserving the original shell's angular momentum and primitive count).
    let shells_json = parsed["shells"].as_array().unwrap();
    let mut shells: Vec<Arc<Shell>> = Vec::with_capacity(shells_json.len());
    for shell in shells_json {
        let l = shell["l"].as_i64().unwrap() as u8;
        let nprim = shell["exponents"].as_array().unwrap().len();
        let exps: Vec<f64> = (0..nprim).map(|_| rng.uniform(0.25, 4.0)).collect();
        let coeffs: Vec<f64> = (0..nprim).map(|_| rng.uniform(0.15, 1.0)).collect();
        let nprim_u = nprim as u16;
        let nctr = 1u16;
        shells.push(Arc::new(
            Shell::try_new(
                0,
                l,
                nprim_u,
                nctr,
                0,
                Representation::Spheric,
                arc_f64(&exps),
                arc_f64(&coeffs),
            )
            .expect("typed AO shell"),
        ));
    }

    // ECP shells — kept VERBATIM (a randomized ECP slab would not be a valid
    // LANL2DZ potential; the randomness lives in the AO basis + geometry).
    let ecp_json = parsed["ecp"]["shells"].as_array().unwrap();
    let mut ecp_shells: Vec<Arc<EcpShell>> = Vec::with_capacity(ecp_json.len());
    for shell in ecp_json {
        let channel_str = shell["channel"].as_str().unwrap();
        let channel = if channel_str == "local" {
            EcpChannel::Local
        } else {
            EcpChannel::Projected(shell["l"].as_i64().unwrap() as u8)
        };
        let r_exps: Vec<i32> = shell["r_exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let radial_power = if r_exps.is_empty() { 0 } else { r_exps[0] } as i16;
        let exps: Vec<f64> = shell["exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let coeffs: Vec<f64> = shell["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let nprim = exps.len() as u16;
        let nctr = 1u16;
        ecp_shells.push(Arc::new(
            EcpShell::try_new(
                0,
                channel,
                radial_power,
                nprim,
                nctr,
                0,
                arc_f64(&exps),
                arc_f64(&coeffs),
            )
            .expect("typed ECP shell"),
        ));
    }

    let basis = BasisSet::try_new_with_ecp(
        atoms,
        Arc::from(shells.clone().into_boxed_slice()),
        Arc::from(ecp_shells.into_boxed_slice()),
    )
    .expect("BasisSet::try_new_with_ecp");
    (basis, shells)
}

/// Evaluate the full `int1e_ecp_sph` AO matrix once via the safe API on the
/// resolved (ROCm) backend. Mirrors `safe_api_ecp_parity.rs::collect_safe_api_ecp_matrix`.
fn collect_ecp_matrix_once(basis: &BasisSet, shells: &[Arc<Shell>]) -> Vec<f64> {
    let shell_nao: Vec<usize> = shells.iter().map(|s| s.ao_per_shell()).collect();
    let n_ao: usize = shell_nao.iter().sum();
    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];
            let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple");
            // ExecutionOptions::default() — the backend is resolved from the
            // CINTX_BACKEND=rocm env var exported by `xtask rocm-oracle`
            // (executor.rs resolve_backend_kind), NOT from backend_intent.
            let request = SessionRequest::new(
                OperatorId::INT1E_ECP_SPH,
                Representation::Spheric,
                basis,
                shell_tuple,
                ExecutionOptions::default(),
            );
            let query = request.query_workspace().expect("query_workspace");
            let output = query.evaluate().expect("evaluate");
            let pair = &output.tensor.owned_values;
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = pair[ii * nj + jj];
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

/// Randomized `int1e_ecp_sph` ROCm idempotency oracle.
///
/// Drives the safe-API ECP path TWICE per random Cu/LANL2DZ system on the ROCm
/// GPU and asserts the two evaluations agree exactly (idempotency,
/// atol=1e-12 / rtol=1e-10), with non-zero output across the suite (proving the
/// `#[cube(launch)]` device kernel ran on the AMD GPU).
#[test]
#[ignore]
fn test_ecp_sph_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;
    let n_cases = 48usize;
    let mut rng = Lcg::new(0x5ec0_ec90_2605_29bf);

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        let (basis, shells) = build_random_ecp_system(&mut rng);

        let out1 = collect_ecp_matrix_once(&basis, &shells);
        let out2 = collect_ecp_matrix_once(&basis, &shells);

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
                    "  rocm ECP RANDOM MISMATCH case {case} idx {idx}: \
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
        "rocm ECP oracle idempotency failed: {mismatch_count} mismatches across {n_cases} random int1e_ecp_sph cases"
    );
    assert!(
        any_nonzero,
        "ECP device output was all zeros across {n_cases} cases — the device kernel did not run on the GPU"
    );
    println!(
        "  PASS: rocm int1e_ecp_sph random idempotency mismatch_count=0 across {n_cases} cases \
         (any_nonzero=true) at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
