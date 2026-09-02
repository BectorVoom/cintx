//! `def2_speed_precision_plan.md` D1.3 and D3.1 — the def2 claims on a GPU
//! backend.
//!
//! # Why a GPU backend is a separate question
//!
//! Every def2 gate so far runs on the CubeCL CPU backend, where `two_e_per_unit`
//! selects the one-quartet-per-unit decomposition. The **cooperative** shape —
//! `per_unit == 0`, one quartet per cube, the cube splitting the contraction,
//! real `sync_cube` barriers — is compiled for every backend and executed on
//! none of them in CI. Two of the plan's claims are only about that shape:
//!
//! * **D1.3** makes the `extended-device-rys` default flip conditional on G2
//!   holding "on the CPU backend *and at least one GPU backend*". The extended
//!   Rys entry is a double-double solver whose correctness rests on the FMA
//!   probe, and both the probe and the solver run on the cooperative shape here
//!   for the first time.
//! * **D3.1** says to run `balanced` tuning where the ranking is a device
//!   timestamp rather than host wall clock — "the condition the module docs name
//!   for turning it on".
//!
//! # The device, and what it is not
//!
//! gfx1151 is an integrated Radeon 860M. Its f64 rate against a 16-core CPU
//! running libcint makes a throughput win implausible, and none is claimed. It
//! is the available **correctness** target for the GPU launch topology, and the
//! available place to ask whether the tuner behaves differently when its
//! rankings come from device timestamps.
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 \
//!   cargo test --release -p cintx-oracle --features cpu,rocm,extended-device-rys \
//!   --test def2_rocm_extended_and_tuning -- --ignored --nocapture
//! ```

#![cfg(all(
    feature = "cpu",
    feature = "rocm",
    feature = "extended-device-rys",
    has_vendor_libcint
))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling, probe_fma_fusion,
};
use cintx_cubecl::tuning::{AutotunePolicy, set_policy, tuned_key_count};
use cintx_cubecl::{BatchShell, evaluate_2e_quartet_batch, prewarm_2e_work_list};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, shell_l, sulfur_dioxide, water};
use std::collections::BTreeSet;

/// Opt-in gate: the ROCm suite needs a real device and is not in the default CI
/// matrix.
fn rocm_requested() -> bool {
    std::env::var("CINTX_ROCM_ORACLE").is_ok_and(|value| value != "0")
}

fn rocm() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Rocm,
        ..Default::default()
    })
    .expect("rocm backend")
}

fn cpu() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// The def2-TZVP quartets whose Rys order is past the polynomial-fit ceiling.
fn high_order_quartets(arrays: &cintx_basis::RawArrays) -> Vec<[u32; 4]> {
    let nbas = arrays.nbas();
    let l = |s: usize| shell_l(arrays, s);
    let mut list = Vec::new();
    for i in 0..nbas {
        for j in 0..=i {
            for k in 0..=i {
                for m in 0..=k {
                    if (l(i) + l(j) + l(k) + l(m)) / 2 + 1 > BASE_DEVICE_NROOTS {
                        list.push([i as u32, j as u32, k as u32, m as u32]);
                    }
                }
            }
        }
    }
    list
}

fn sph_len(arrays: &cintx_basis::RawArrays, quartet: [u32; 4]) -> usize {
    use cintx_basis::raw::{BAS_SLOTS, NCTR_OF};
    quartet
        .iter()
        .map(|&s| {
            (2 * shell_l(arrays, s as usize) + 1)
                * arrays.bas[s as usize * BAS_SLOTS + NCTR_OF] as usize
        })
        .product()
}

/// **D1.3, first half.** The FMA-fusion probe passes on ROCm, so the extended
/// ceiling is live there — the precondition for everything below, and the thing
/// the default flip is conditional on.
///
/// A backend whose probe *failed* would keep the base ceiling by design, and
/// that is a reported verification status rather than a bug. This test records
/// which case gfx1151 is in.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn rocm_fma_probe_and_extended_ceiling() {
    if !rocm_requested() {
        eprintln!("skipped: set CINTX_ROCM_ORACLE=1");
        return;
    }
    let backend = rocm();
    let probe = probe_fma_fusion(&backend);
    println!(
        "rocm FMA probe: target={:?} fused={} divergent={}/{}",
        probe.target, probe.fused, probe.divergent, probe.pairs
    );
    assert!(
        probe.fused,
        "the double-double Wheeler path needs a true fused multiply-add; this \
         backend lowered `fma` to a multiply followed by an add, so its ceiling \
         stays at {BASE_DEVICE_NROOTS} by design and the extended tests below \
         cannot run"
    );
    for family in [
        RysFamily::Int2e,
        RysFamily::Int3c2e,
        RysFamily::Int3c2eDeriv,
        RysFamily::Int2c2e,
        RysFamily::Int1e,
        RysFamily::Int1eDeriv,
    ] {
        assert_eq!(
            device_nroots_ceiling(&backend, family),
            EXTENDED_DEVICE_NROOTS,
            "{} must reach the extended ceiling on a backend whose probe passed",
            family.name()
        );
    }
}

/// **D1.3, second half, and the one that matters.** The extended Rys path
/// produces vendor-compatible def2-TZVP results *on the cooperative launch
/// shape*.
///
/// Two comparisons, because they answer different questions:
///
/// 1. **ROCm vs vendored libcint** — is the answer right?
/// 2. **ROCm vs the CPU backend, in this process** — did the launch topology
///    change it? The two decompositions run the same arithmetic in a different
///    order, so this is a tolerance rather than bit-identity, but a divergence
///    beyond a few ulp would mean the cooperative shape is doing something the
///    per-unit one is not.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn rocm_extended_rys_matches_vendor_on_def2_tzvp() {
    if !rocm_requested() {
        eprintln!("skipped: set CINTX_ROCM_ORACLE=1");
        return;
    }
    // The extended path's double-double arms round the last f64 bit differently
    // from the vendor's 80-bit `long double`; these are the floors every
    // extended-Rys gate uses.
    const ATOL: f64 = 1e-11;
    const RTOL: f64 = 1e-9;

    for (label, molecule) in [
        ("H2O", water(StandardBasis::Def2Tzvp)),
        ("SO2", sulfur_dioxide(StandardBasis::Def2Tzvp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let shells = batch_shells(&arrays);
        let list = high_order_quartets(&arrays);
        assert!(
            !list.is_empty(),
            "{label}/def2-TZVP produced no class past nroots={BASE_DEVICE_NROOTS}"
        );

        let device = evaluate_2e_quartet_batch(&rocm(), &shells, &list)
            .unwrap_or_else(|e| panic!("{label}: rocm high-order 2e batch failed: {e}"));
        let host = evaluate_2e_quartet_batch(&cpu(), &shells, &list)
            .unwrap_or_else(|e| panic!("{label}: cpu high-order 2e batch failed: {e}"));

        let mut orders: BTreeSet<usize> = BTreeSet::new();
        let (mut vendor_worst, mut cross_worst) = (0.0_f64, 0.0_f64);
        let (mut vendor_bad, mut compared) = (0_usize, 0_usize);

        for (index, &q) in list.iter().enumerate() {
            let len = sph_len(&arrays, q);
            let mut expected = vec![0.0_f64; len];
            vendor_ffi::vendor_int2e_sph(
                &mut expected,
                &[q[0] as i32, q[1] as i32, q[2] as i32, q[3] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            orders.insert(
                q.iter()
                    .map(|&s| shell_l(&arrays, s as usize))
                    .sum::<usize>()
                    / 2
                    + 1,
            );
            let d = &device.values[device.offsets[index]..device.offsets[index] + len];
            let h = &host.values[host.offsets[index]..host.offsets[index] + len];
            for ((e, a), c) in expected.iter().zip(d).zip(h) {
                compared += 1;
                let tol = ATOL.max(RTOL * e.abs());
                let diff = (e - a).abs();
                vendor_worst = vendor_worst.max(diff / tol);
                if diff > tol {
                    vendor_bad += 1;
                    if vendor_bad <= 5 {
                        eprintln!(
                            "  MISMATCH {label} {q:?}: vendor={e:.15e} rocm={a:.15e} \
                             |d|={diff:.3e} tol={tol:.3e}"
                        );
                    }
                }
                cross_worst = cross_worst.max((a - c).abs());
            }
        }

        println!(
            "{label}/def2-TZVP on rocm: quartets={} elements={compared} nroots={orders:?}  \
             worst |diff|/tol vs vendor={vendor_worst:.3}  worst |rocm-cpu|={cross_worst:.3e}",
            list.len()
        );
        assert_eq!(
            vendor_bad, 0,
            "{label}: {vendor_bad} of {compared} rocm extended-Rys elements exceeded \
             max(atol={ATOL:e}, rtol={RTOL:e})"
        );
        assert!(
            cross_worst <= ATOL,
            "{label}: the cooperative and per-unit launch shapes disagreed by \
             {cross_worst:.3e}; the launch topology must not change a result"
        );
    }
}

/// **D3.1.** `balanced` tuning over the def2 work lists, on a backend where the
/// tuner's ranking comes from device timestamps rather than host wall clock.
///
/// # What this reports, and what it does not
///
/// It does *not* assert a speed-up. The plan's D3 exit is "Phase 6 exit gates
/// hold on the def2 work lists on at least one GPU backend, **or the measured
/// no-win is documented per backend the way the CPU result was**" — and on an
/// integrated GPU against a 16-core CPU the second is the likely outcome. What
/// it asserts is the part that must hold either way: **tuning must not change a
/// value.** The tuner picks a launch geometry; the kernel covers the same index
/// space at every geometry, so a difference here is a bug, not a trade-off.
///
/// The timings are printed for the record. `--nocapture` is what makes this
/// test worth running.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn rocm_balanced_tuning_over_def2_work_lists() {
    if !rocm_requested() {
        eprintln!("skipped: set CINTX_ROCM_ORACLE=1");
        return;
    }
    let repeats = 5;

    for (label, molecule) in [
        ("H2O/def2-SVP", water(StandardBasis::Def2Svp)),
        ("SO2/def2-SVP", sulfur_dioxide(StandardBasis::Def2Svp)),
        ("H2O/def2-TZVP", water(StandardBasis::Def2Tzvp)),
    ] {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
        let quartets = enumerate_quartets(&enumerate_pairs(&basis));
        let list: Vec<[u32; 4]> = quartets
            .iter()
            .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
            .collect();
        let shells: Vec<BatchShell> = batch_shells(&arrays);
        let backend = rocm();

        let measure = |policy: AutotunePolicy| -> (f64, Vec<f64>) {
            set_policy(policy);
            // Compilation is per (policy, geometry); warming inside the policy
            // being measured is what keeps the timing a steady-state one.
            prewarm_2e_work_list(&backend, &shells, &list).expect("prewarm");
            let mut best = f64::INFINITY;
            let mut values = Vec::new();
            for _ in 0..repeats {
                let start = std::time::Instant::now();
                let output = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("batch");
                best = best.min(start.elapsed().as_secs_f64());
                values = output.values;
            }
            (best, values)
        };

        let (off_secs, off_values) = measure(AutotunePolicy::Off);
        let keys_before = tuned_key_count();
        let (tuned_secs, tuned_values) = measure(AutotunePolicy::Balanced);
        let keys_after = tuned_key_count();
        set_policy(AutotunePolicy::Off);

        let worst = off_values
            .iter()
            .zip(&tuned_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "{label:<14} quartets={:<7} off={off_secs:.5} s  balanced={tuned_secs:.5} s  \
             ({:.2}x)  tuned keys {keys_before}->{keys_after}  worst |off-tuned|={worst:.3e}",
            list.len(),
            off_secs / tuned_secs.max(f64::MIN_POSITIVE),
        );

        assert_eq!(
            off_values.len(),
            tuned_values.len(),
            "{label}: tuning changed the output length"
        );
        assert_eq!(
            worst, 0.0,
            "{label}: tuning changed a value by {worst:.3e}. The kernel covers the \
             same index space at every geometry, so `cube_dim` buys speed and never \
             results — a difference here is a bug, not a trade-off"
        );
    }
}
