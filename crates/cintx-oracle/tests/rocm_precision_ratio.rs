//! Part 6 — the f64:f32 arithmetic-rate ratio on the ROCm device.
//!
//! This is the plan's carried-forward unknown: every tiering idea (run the Rys
//! roots in f32 and refine, screen in f32, split a work list by required
//! accuracy) is a guess until the ratio is measured on the actual target. On a
//! discrete HPC part it is 1:2; on a consumer or integrated part it is commonly
//! 1:16 or 1:32, and at 1:32 a tiering scheme that pays a conversion and a
//! refinement pass is not obviously ahead of running f64 throughout.
//!
//! The measurement is deliberately *not* an integral benchmark — see
//! `cintx_cubecl::precision_ratio` for what the number does and does not cover.
//! It reports rather than asserts a bound: what the right ratio "should" be is
//! exactly the thing that was unknown, so an assertion would be a guess dressed
//! as a gate. The CPU backend is measured in the same run as a control.
//!
//! ```text
//! CINTX_ROCM_ORACLE=1 cargo test --release -p cintx-oracle \
//!   --features cpu,rocm --test rocm_precision_ratio -- --ignored --nocapture
//! ```

#![cfg(all(feature = "cpu", feature = "rocm"))]

use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::measure_precision_ratio;
use cintx_runtime::{BackendIntent, BackendKind};

/// Opt-in gate: this needs a real device and is not part of the default CI
/// matrix.
fn rocm_requested() -> bool {
    std::env::var("CINTX_ROCM_ORACLE").is_ok_and(|value| value != "0")
}

/// Measure one backend at the current probe size and print a row.
fn report(label: &str, kind: BackendKind) {
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: kind,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{label} backend: {error}"));

    let measured = measure_precision_ratio(&backend);
    let fmas = measured.work_items as f64 * f64::from(measured.iterations);
    println!(
        "  {label:<6} {:>8} items x {} FMAs   f64 {:>9.3} ms ({:>7.2} GFMA/s)   \
         f32 {:>9.3} ms ({:>7.2} GFMA/s)   ratio {:>5.2}x",
        measured.work_items,
        measured.iterations,
        measured.f64_ns as f64 / 1e6,
        fmas / measured.f64_ns.max(1) as f64,
        measured.f32_ns as f64 / 1e6,
        fmas / measured.f32_ns.max(1) as f64,
        measured.ratio,
    );

    // The only thing worth asserting is that the probe ran: a zero or
    // non-finite ratio means the launch did not execute, which would make the
    // printed number a fiction. No bound is asserted on the ratio itself —
    // what it "should" be is precisely what was unknown.
    assert!(
        measured.f64_ns > 0 && measured.f32_ns > 0,
        "{label}: probe did not run"
    );
    assert!(
        measured.ratio.is_finite() && measured.ratio > 0.0,
        "{label}: degenerate ratio"
    );
}

#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn rocm_f64_to_f32_arithmetic_ratio() {
    if !rocm_requested() {
        eprintln!("CINTX_ROCM_ORACLE unset — skipping the ROCm precision-ratio probe");
        return;
    }

    println!("\n{:=<104}", "");
    println!("f64 : f32 arithmetic-rate ratio (dependent FMA chain, no memory traffic)");
    println!("{:=<104}", "");

    report("cpu", BackendKind::Cpu);

    // The ratio is swept over occupancy rather than reported at one point,
    // because the two ends mean different things and an integral kernel lives
    // at both: a short work list is latency-bound (the recurrence chain is the
    // critical path, and a dependent FMA chain is exactly that shape), while a
    // Fock-sized list saturates the device.
    for items in [2048_usize, 8192, 65_536, 262_144] {
        // SAFETY: single-threaded test; the probe reads this once per call and
        // nothing else in this binary looks at the variable.
        unsafe { std::env::set_var("CINTX_PRECISION_PROBE_ITEMS", items.to_string()) };
        report("rocm", BackendKind::Rocm);
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var("CINTX_PRECISION_PROBE_ITEMS") };

    println!();
    println!("  Reading: the ratio is the ceiling on what a precision-tiering scheme could");
    println!("  recover, and only on the *arithmetic* share of a kernel — not on its");
    println!("  transcendental calls or its memory traffic. It rises with occupancy because");
    println!("  f64 saturates first: at the low end the chain is latency-bound and both");
    println!("  precisions wait, at the high end f64 is rate-limited and f32 is not.");
    println!();
}
