//! Measuring a backend's f64-to-f32 throughput ratio (Part 6).
//!
//! Every tiering decision this project might make — run the Rys roots in f32
//! and refine, keep a mixed-precision screening pass, split a work list by
//! required accuracy — rests on one number nobody has measured: how much slower
//! f64 is than f32 on the target device. On a discrete HPC GPU it is 1:2; on a
//! consumer or integrated part it is commonly 1:16 or 1:32, and at 1:32 a
//! tiering scheme that pays a conversion and a refinement pass is not obviously
//! ahead of just running f64.
//!
//! # What is measured
//!
//! A dependent chain of fused multiply-adds, one per work item, with no memory
//! traffic between iterations and no way to constant-fold it: each iteration's
//! input is the previous iteration's output, and the seed depends on the work
//! item's index. The two precisions run the *same* kernel source through the
//! same launch geometry, so the ratio is the arithmetic-rate ratio and not a
//! difference in occupancy, unrolling or memory behaviour.
//!
//! # What is not measured
//!
//! Not integral throughput. A real kernel is a mix of arithmetic, transcendental
//! calls and memory traffic, and only the first of those scales with this ratio.
//! Read it as an upper bound on what a tiering scheme could recover, not as a
//! prediction of what it would.

// The `as u32` casts on the CubeCL builtins are load-bearing under `#[cube]` —
// `CUBE_POS` and friends expand to `NativeExpand<u32>` and the uniform cast form
// is what the rest of this crate uses. Clippy sees the post-expansion type.
#![allow(clippy::unnecessary_cast)]

use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient};
use std::time::Instant;

use crate::backend::ResolvedBackend;

/// One backend's measured f64:f32 arithmetic-rate ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecisionRatio {
    /// Best-of wall-clock for the f64 launch, in nanoseconds.
    pub f64_ns: u64,
    /// Best-of wall-clock for the f32 launch, in nanoseconds.
    pub f32_ns: u64,
    /// `f64_ns / f32_ns`. `1.0` means parity; `16.0` means f64 is 16x slower.
    pub ratio: f64,
    /// Work items per launch.
    pub work_items: usize,
    /// Dependent FMAs per work item.
    pub iterations: u32,
}

/// A dependent FMA chain, one per work item.
///
/// `acc` feeds itself every iteration, so the chain cannot be vectorized away
/// or hoisted, and the seed depends on the work item's index, so it cannot be
/// constant-folded. The single store at the end keeps the result live.
#[cube(launch, launch_unchecked)]
fn precision_probe_kernel<F: Float + CubeElement>(out: &mut Array<F>, seed: F, iterations: u32) {
    let slot = (CUBE_POS as u32) * (CUBE_DIM as u32) + (UNIT_POS as u32);
    if (slot as usize) < out.len() {
        let mut acc = seed + F::cast_from(slot);
        let mul = F::new(1.000_000_1_f32);
        let add = F::new(0.999_999_9_f32);
        let mut k = 0u32;
        while k < iterations {
            acc = acc * mul + add;
            k += 1u32;
        }
        out[slot as usize] = acc;
    }
}

/// Work items per launch. Large enough to fill a GPU, small enough that the
/// CPU backend finishes in well under a second.
///
/// `CINTX_PRECISION_PROBE_ITEMS` overrides it. Sweeping it is how a reader
/// tells a latency-bound measurement from a throughput-bound one: if the ratio
/// moves with occupancy, the chain was not deep enough to hide latency.
const DEFAULT_WORK_ITEMS: usize = 8192;
/// Dependent FMAs per work item. `CINTX_PRECISION_PROBE_ITERS` overrides it.
const DEFAULT_ITERATIONS: u32 = 4096;
/// Timed launches; the best is reported, so a scheduling hiccup cannot inflate
/// the ratio.
const REPEATS: usize = 5;

fn work_items() -> usize {
    std::env::var("CINTX_PRECISION_PROBE_ITEMS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_WORK_ITEMS)
}

fn iterations() -> u32 {
    std::env::var("CINTX_PRECISION_PROBE_ITERS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_ITERATIONS)
}

/// Time one precision's launch, best of [`REPEATS`], after one warm-up.
fn time_one<R: Runtime, F: Float + CubeElement>(client: &ComputeClient<R>) -> u64 {
    let items = work_items();
    let iters = iterations();
    let cube_dim = CubeDim::new_1d(64);
    let cubes = items.div_ceil(64) as u32;
    let out = client.empty(items * std::mem::size_of::<F>());

    let launch = || {
        // SAFETY: `out` is allocated at exactly `items` elements and the
        // kernel bounds its own index against `out.len()`.
        unsafe {
            precision_probe_kernel::launch_unchecked::<F, R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                cube_dim,
                ArrayArg::from_raw_parts(out.clone(), items),
                F::from_int(1),
                iters,
            );
        }
        // The read is what forces the device to finish; without it the timing
        // would measure submission, not execution.
        let _ = client.read_one_unchecked(out.clone());
    };

    launch();
    let mut best = u64::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        launch();
        best = best.min(start.elapsed().as_nanos() as u64);
    }
    best
}

/// Measure `backend`'s f64:f32 arithmetic-rate ratio.
///
/// See the module docs for what the number does and does not mean.
pub fn measure_precision_ratio(backend: &ResolvedBackend) -> PrecisionRatio {
    let (f64_ns, f32_ns) = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => (
            time_one::<cubecl::cpu::CpuRuntime, f64>(client),
            time_one::<cubecl::cpu::CpuRuntime, f32>(client),
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => (
            time_one::<cubecl_wgpu::WgpuRuntime, f64>(client),
            time_one::<cubecl_wgpu::WgpuRuntime, f32>(client),
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => (
            time_one::<cubecl_cuda::CudaRuntime, f64>(client),
            time_one::<cubecl_cuda::CudaRuntime, f32>(client),
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => (
            time_one::<cubecl_hip::HipRuntime, f64>(client),
            time_one::<cubecl_hip::HipRuntime, f32>(client),
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => (
            time_one::<cubecl_wgpu::WgpuRuntime, f64>(client),
            time_one::<cubecl_wgpu::WgpuRuntime, f32>(client),
        ),
    };

    PrecisionRatio {
        f64_ns,
        f32_ns,
        ratio: f64_ns as f64 / f32_ns.max(1) as f64,
        work_items: work_items(),
        iterations: iterations(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe runs and reports a finite, positive ratio on the CPU backend.
    ///
    /// No bound is asserted: on a CPU the two precisions are both one FMA per
    /// cycle per lane and the ratio is near 1, but SIMD width doubles for f32,
    /// so the honest expectation is "somewhere between 1 and 2" and pinning it
    /// tighter would make this a flaky test of the host's vectorizer.
    #[cfg(feature = "cpu")]
    #[test]
    fn precision_probe_reports_a_finite_ratio_on_cpu() {
        use cintx_runtime::{BackendIntent, BackendKind};
        let backend = ResolvedBackend::from_intent(&BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend");
        let measured = measure_precision_ratio(&backend);
        assert!(measured.f64_ns > 0);
        assert!(measured.f32_ns > 0);
        assert!(
            measured.ratio.is_finite() && measured.ratio > 0.0,
            "ratio must be finite and positive, got {}",
            measured.ratio
        );
    }
}
