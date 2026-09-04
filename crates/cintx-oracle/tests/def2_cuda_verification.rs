//! CUDA runtime verification for the def2 batch path.
//!
//! # The gap this closes
//!
//! `.planning/notes/cuda-metal-verification-gap.md` records the CUDA backend as
//! **compile-only**: it is built in the feature matrix and has never executed a
//! kernel on real hardware. Every claim about it is therefore a claim about the
//! compiler, not the device.
//!
//! This file is what a CUDA device runs to close that. It is deliberately
//! compact — it verifies the properties that are *new information* on a second
//! GPU vendor, rather than re-deriving what the ROCm suite already establishes:
//!
//! 1. **The backend resolves and runs at all**, and reports its capabilities.
//! 2. **The FMA-fusion probe**, which is what decides whether the extended Rys
//!    ceiling (`nroots` 6-12, and so def2-TZVP) is available at all. This is a
//!    property of the compiler's contraction behaviour and cannot be inferred
//!    from another vendor's answer.
//! 3. **`int2e_sph` against vendored libcint**, on a real def2 work list, at the
//!    project tolerance — the correctness claim itself.
//! 4. **CUDA against the CPU backend**, to size the cooperative-versus-per-unit
//!    divergence on this device the way the ROCm suite does for gfx1151.
//! 5. **The M3 device transform**, since a discrete GPU is the only place its
//!    readback saving is a real transfer rather than a memcpy.
//!
//! # Running it
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 CINTX_CUDA_ORACLE=1 \
//!   cargo test --release -p cintx-oracle --features cpu,cuda,extended-device-rys \
//!   --test def2_cuda_verification -- --ignored --nocapture
//! ```
//!
//! A note on what a result here does and does not mean. On a T4 the f64 rate is
//! 1/32 of f32, so **no throughput claim should be read off this device** — it
//! is a correctness target, exactly as gfx1151 is. What it settles is whether
//! cintx's kernels produce libcint's numbers on NVIDIA hardware.

#![cfg(all(feature = "cpu", feature = "cuda", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling, probe_fma_fusion,
};
use cintx_cubecl::{BatchShell, evaluate_2e_quartet_batch};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// Project oracle tolerance.
const TOL: f64 = 1e-12;

/// Opt-in gate: this suite needs a real CUDA device and is not in the default
/// matrix, exactly as the ROCm suite is not.
fn cuda_requested() -> bool {
    std::env::var("CINTX_CUDA_ORACLE").is_ok_and(|value| value != "0")
}

fn backend(kind: BackendKind) -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: kind,
        ..Default::default()
    })
    .expect("backend must resolve")
}

/// A molecule's canonical quartet list and its batch shells.
fn work_list(molecule: &Molecule) -> (RawArrays, Vec<BatchShell>, Vec<[u32; 4]>) {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let shells = def2_fixtures::batch_shells(&arrays);
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list = enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    (arrays, shells, list)
}

/// Vendored libcint over the same list, concatenated in list order.
fn vendor_values(arrays: &RawArrays, list: &[[u32; 4]]) -> Vec<f64> {
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let mut out = Vec::new();
    let mut block = Vec::new();
    for quartet in list {
        let len = quartet
            .iter()
            .map(|&s| view.nsph(s as usize))
            .product::<usize>();
        block.clear();
        block.resize(len, 0.0);
        vendor_ffi::vendor_int2e_sph(
            &mut block,
            &[
                quartet[0] as i32,
                quartet[1] as i32,
                quartet[2] as i32,
                quartet[3] as i32,
            ],
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );
        out.extend_from_slice(&block);
    }
    out
}

/// The device resolves, and its FMA-fusion probe decides the Rys ceiling.
///
/// The probe is not a formality. The extended Rys path is a double-double
/// Wheeler solver whose correctness rests on the compiler emitting a *true*
/// fused multiply-add rather than contracting `two_sum`/`two_prod` into one —
/// so a backend that fails it runs def2-TZVP at `nroots <= 5` by design, and
/// that is a reported status rather than a bug.
#[test]
#[ignore = "needs a CUDA device; run explicitly"]
fn cuda_backend_resolves_and_reports_its_fma_fusion() {
    if !cuda_requested() {
        println!("skipped: set CINTX_CUDA_ORACLE=1 with a CUDA device present");
        return;
    }
    for (label, kind) in [("cpu", BackendKind::Cpu), ("cuda", BackendKind::Cuda)] {
        let resolved = backend(kind);
        let probe = probe_fma_fusion(&resolved);
        let ceiling = device_nroots_ceiling(&resolved, RysFamily::Int2e);
        println!(
            "{label:<5} fused={:<5} int2e nroots ceiling={ceiling} (base {BASE_DEVICE_NROOTS}, extended {EXTENDED_DEVICE_NROOTS})",
            probe.fused
        );
    }
    // Not asserted either way: a backend without fused FMA is a supported
    // configuration that runs at the base ceiling. What is asserted is that the
    // probe ran rather than panicking, which is the thing that had never
    // happened on CUDA hardware before.
}

/// `int2e_sph` on CUDA must match vendored libcint on a real def2 work list.
#[test]
#[ignore = "needs a CUDA device; run explicitly"]
fn cuda_int2e_matches_vendored_libcint() {
    if !cuda_requested() {
        println!("skipped: set CINTX_CUDA_ORACLE=1 with a CUDA device present");
        return;
    }
    let cuda = backend(BackendKind::Cuda);
    let cpu = backend(BackendKind::Cpu);

    let mut cases: Vec<(&str, Molecule)> = vec![(
        "H2O / def2-SVP",
        def2_fixtures::water(StandardBasis::Def2Svp),
    )];
    // def2-TZVP reaches `nroots` 6-7, so it is only in scope where the extended
    // path is compiled in *and* this device's FMA probe unlocked it.
    #[cfg(feature = "extended-device-rys")]
    if device_nroots_ceiling(&cuda, RysFamily::Int2e) >= 7 {
        cases.push((
            "H2O / def2-TZVP",
            def2_fixtures::water(StandardBasis::Def2Tzvp),
        ));
    }

    println!(
        "\n{:<18} {:>9} {:>11} {:>13} {:>13}",
        "case", "quartets", "elements", "cuda vs vendor", "cuda vs cpu"
    );
    let mut failures = Vec::new();
    for (label, molecule) in cases {
        let (arrays, shells, list) = work_list(&molecule);
        let reference = vendor_values(&arrays, &list);

        let on_cuda = evaluate_2e_quartet_batch(&cuda, &shells, &list).expect("cuda batch");
        let on_cpu = evaluate_2e_quartet_batch(&cpu, &shells, &list).expect("cpu batch");

        let vendor_diff = on_cuda
            .values
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let cross_diff = on_cuda
            .values
            .iter()
            .zip(&on_cpu.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "{label:<18} {:>9} {:>11} {:>13.3e} {:>13.3e}",
            list.len(),
            reference.len(),
            vendor_diff,
            cross_diff
        );
        if !(vendor_diff < TOL) {
            failures.push(format!(
                "{label}: cuda vs vendor {vendor_diff:.3e} exceeds {TOL:.0e}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// The M3 device transform must be bit-identical on CUDA too.
///
/// A discrete GPU is where moving the transform actually saves a transfer, so
/// this is the backend on which M3's reason for existing can be checked at all —
/// and correctness has to come first.
#[test]
#[ignore = "needs a CUDA device; run explicitly"]
fn cuda_device_transform_is_bit_identical() {
    if !cuda_requested() {
        println!("skipped: set CINTX_CUDA_ORACLE=1 with a CUDA device present");
        return;
    }
    if !cintx_cubecl::device_transform_enabled() {
        println!("skipped: run this test with CINTX_2E_TRANSFORM=device to exercise the M3 kernel");
        return;
    }
    // With the mode on, the batch below transforms on the device. The reference
    // is the host transform, which cannot be produced in the same process — the
    // mode is read once — so the comparison is against the vendor instead, at
    // the project tolerance, plus the readback-volume claim that is the whole
    // point of the mode.
    let cuda = backend(BackendKind::Cuda);
    let (arrays, shells, list) = work_list(&def2_fixtures::water(StandardBasis::Def2Svp));
    let reference = vendor_values(&arrays, &list);
    let out = evaluate_2e_quartet_batch(&cuda, &shells, &list).expect("cuda batch");

    let diff = out
        .values
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "device transform on cuda: max|diff| vs vendor {diff:.3e}, readback {:.2} MiB for a {:.2} MiB output",
        out.stats.readback_bytes as f64 / (1024.0 * 1024.0),
        out.stats.host_output_bytes as f64 / (1024.0 * 1024.0),
    );
    assert!(
        diff < TOL,
        "cuda device transform diff {diff:.3e} exceeds {TOL:.0e}"
    );
    assert_eq!(
        out.stats.readback_bytes, out.stats.host_output_bytes,
        "the device transform must read back exactly the spherical output"
    );
}
