//! S3 — the cooperative decomposition's G build, exercised without a GPU.
//!
//! # What this covers, and why it can run in CI
//!
//! `two_electron_scalar_kernel` compiles two shapes. The **per-unit** one
//! (`per_unit == 1`) gives each work item a whole quartet and is what the
//! CubeCL CPU runtime wants; it is what every def2 and GTH gate measures. The
//! **cooperative** one (`per_unit == 0`) puts one quartet on a cube and splits
//! the work across its lanes, and it is what every GPU backend runs. Until S3
//! the whole VRR/HRR G build sat inside a `lane == 0` region there, so the
//! other lanes idled through it.
//!
//! S3 hands each lane a slice of the build. The slices are the `(axis, root)`
//! pairs: `build_2e_shape` makes every stride (`di`, `dk`, `dl`, `dj`, and so
//! the VRR's `g2d_ijmax`/`g2d_klmax`) a multiple of `nroots` in a root-fastest
//! layout, so the recurrences never cross a root, and `off = gx_off + axis *
//! g_size` keeps the axes apart. Each element is therefore still computed by
//! exactly the expression that computed it before, on a different lane.
//!
//! **So the gate is bit-identity, not a divergence budget** — and it is
//! checkable on the CPU backend, where both shapes can be pinned inside one
//! process. That matters here: gfx1151 is this host's display GPU, and a long
//! compute dispatch on it resets the device and takes the desktop session with
//! it (`gth_molopt_speed_memory_plan.md` §8.6). A GPU is where S3 *pays*; it
//! is not where S3 has to be *checked*.
//!
//! The cooperative arm is deliberately slow on the CPU runtime — a unit is an
//! OS thread and `sync_cube` a global spin barrier — so the work lists here
//! are small and the cube is pinned narrow. This is a correctness vehicle and
//! never a timing one.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{RawArrays, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    BatchShell, ResidentTwoEBasis, evaluate_2e_quartet_batch_resident, set_cooperative_build_split,
    set_two_e_cube_dim, set_two_e_per_unit,
};
use cintx_driver::{BasisView, bucket_quartets, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, water};

/// Lanes per cube in the pinned cooperative runs.
///
/// Four, not the heuristic width: on the CubeCL CPU runtime the cube dimension
/// *is* the OS-thread count, and every `sync_cube` is a spin barrier across all
/// of them. Four is enough for the lane split to be real — with `nroots >= 2`
/// every lane owns at least one `(axis, root)` slice, and with `nroots == 1`
/// three of the four do — while staying cheap enough to run in CI.
const COOPERATIVE_LANES: u32 = 4;

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// A short work list that still spans many angular-momentum classes: one
/// quartet from each launch class, lowest Rys orders first, capped.
///
/// Coverage matters more than length here. A long list of `(ss|ss)` would
/// exercise one `nroots` and one HRR branch; one quartet from each class
/// reaches all four comptime HRR branches and several Rys orders, which is
/// where a wrong lane-ownership map would show.
fn one_quartet_per_class(arrays: &RawArrays, max_nroots: u32, cap: usize) -> Vec<[u32; 4]> {
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&view));
    let mut buckets = bucket_quartets(&view, &quartets);
    buckets.sort_by_key(|bucket| bucket.class.nroots);
    buckets
        .iter()
        .filter(|bucket| bucket.class.nroots <= max_nroots)
        .filter_map(|bucket| bucket.quartets.first())
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .take(cap)
        .collect()
}

/// Evaluate `list` under a pinned decomposition, restoring the default after.
fn evaluate_pinned(
    shells: &[BatchShell],
    list: &[[u32; 4]],
    per_unit: bool,
    split_build: bool,
) -> (Vec<f64>, Vec<usize>) {
    let backend = cpu_backend();
    set_two_e_per_unit(Some(per_unit));
    set_cooperative_build_split(split_build);
    set_two_e_cube_dim(if per_unit {
        None
    } else {
        Some(COOPERATIVE_LANES)
    });
    // The residency is tagged by backend, not by decomposition, but it is
    // rebuilt per arm anyway so neither arm can inherit the other's device
    // buffers and quietly agree for the wrong reason.
    let resident = ResidentTwoEBasis::new(&backend, shells).expect("residency");
    let out = evaluate_2e_quartet_batch_resident(&backend, &resident, list).expect("2e batch");
    set_two_e_per_unit(None);
    set_two_e_cube_dim(None);
    set_cooperative_build_split(true);
    (out.values, out.offsets)
}

/// Vendored libcint over the same list, laid out at `offsets`.
fn vendor_values(
    arrays: &RawArrays,
    list: &[[u32; 4]],
    offsets: &[usize],
    total: usize,
) -> Vec<f64> {
    let mut vendor = vec![0.0_f64; total];
    let mut scratch = vec![0.0_f64; 8192];
    for (index, quartet) in list.iter().enumerate() {
        let start = offsets[index];
        let end = offsets.get(index + 1).copied().unwrap_or(total);
        let len = end - start;
        if scratch.len() < len {
            scratch.resize(len, 0.0);
        }
        vendor_ffi::vendor_int2e_sph(
            &mut scratch[..len],
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
        vendor[start..end].copy_from_slice(&scratch[..len]);
    }
    vendor
}

/// Run both arms over `list` and hold them to bit-identity and to the vendor.
fn assert_arms_agree(label: &str, arrays: &RawArrays, list: &[[u32; 4]]) {
    assert!(!list.is_empty(), "{label}: empty work list");
    let shells = batch_shells(arrays);

    let (per_unit, offsets) = evaluate_pinned(&shells, list, true, true);
    let (cooperative, coop_offsets) = evaluate_pinned(&shells, list, false, true);
    // The pre-S3 shape, on the same cube: lane 0 builds, the rest wait. It is
    // the A/B reference every S3 timing claim is measured against, so it has
    // to be held to the same values, not merely kept compiling.
    let (lane0, lane0_offsets) = evaluate_pinned(&shells, list, false, false);
    assert_eq!(offsets, coop_offsets, "{label}: block layout");
    assert_eq!(offsets, lane0_offsets, "{label}: block layout (lane0)");
    assert_eq!(per_unit.len(), cooperative.len(), "{label}: output length");
    let lane0_differing = lane0
        .iter()
        .zip(&cooperative)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        lane0_differing, 0,
        "{label}: {lane0_differing} elements differ between the lane-0 and split G builds          on the same cube — the A/B the S3 speed claim rests on"
    );

    let mut differing = 0_usize;
    let mut worst = 0.0_f64;
    let mut first = String::new();
    for (index, (a, b)) in per_unit.iter().zip(&cooperative).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        differing += 1;
        worst = worst.max((a - b).abs());
        if first.is_empty() {
            first = format!("element {index}: per-unit {a:.17e} cooperative {b:.17e}");
        }
    }
    assert_eq!(
        differing,
        0,
        "{label}: {differing} of {} elements differ between the per-unit and cooperative \
         decompositions (max |diff| {worst:.3e}). S3 partitions the G build by (axis, root), \
         and every stride is a multiple of nroots in a root-fastest layout, so each element \
         must still be computed by exactly its own expression. First {first}",
        per_unit.len()
    );

    // Both arms against a third party, so a shared error cannot pass as
    // agreement.
    let vendor = vendor_values(arrays, list, &offsets, per_unit.len());
    for (name, values) in [("per-unit", &per_unit), ("cooperative", &cooperative)] {
        let mut over = 0_usize;
        let mut worst = 0.0_f64;
        for (v, a) in vendor.iter().zip(values.iter()) {
            let diff = (v - a).abs();
            worst = worst.max(diff);
            if diff > 1e-12 {
                over += 1;
            }
        }
        assert_eq!(
            over, 0,
            "{label}: the {name} arm has {over} elements over 1e-12 against vendored \
             libcint (max |diff| {worst:.3e})"
        );
        println!("  {label:<34} {name:<12} vs vendor: max|diff|={worst:.3e}");
    }
}

/// def2-SVP: segmented, so the contraction is the fast path and the G build is
/// all the cooperative arm has to split.
#[test]
fn cooperative_g_build_is_bit_identical_on_def2() {
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let list = one_quartet_per_class(&arrays, 3, 24);
    println!("\ndef2-SVP water: {} quartets, one per class", list.len());
    assert_arms_agree("H2O / def2-SVP", &arrays, &list);
}

/// GTH-MOLOPT: generally contracted, so the cooperative arm runs the staged
/// contraction's four stages beside the split G build. The two features were
/// written independently and this is the only gate that crosses them.
#[cfg(feature = "gth")]
#[test]
fn cooperative_g_build_is_bit_identical_on_gth() {
    for (label, arrays) in def2_fixtures::gth_workloads() {
        if !label.starts_with("H2O") {
            continue;
        }
        let list = one_quartet_per_class(&arrays, 3, 12);
        println!("\n{label}: {} quartets, one per class", list.len());
        assert_arms_agree(&label, &arrays, &list);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S3's timing question, which only a GPU can answer
// ─────────────────────────────────────────────────────────────────────────────

/// Every canonical quartet of a basis.
#[cfg(feature = "rocm")]
fn full_list(arrays: &RawArrays) -> Vec<[u32; 4]> {
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect()
}

/// Split versus lane-0 G build on ROCm, alternated inside one process.
///
/// The CPU backend cannot answer this. There a cube unit is an OS thread and
/// `sync_cube` a global spin barrier, so widening the cube to parallelise the
/// build costs far more than the build; the cooperative arm exists there to be
/// *checked*, which the tests above do. On a GPU the barrier is a workgroup
/// barrier and the lanes are real, which is the shape S3 was written for.
///
/// Interleaved and best-of, for the reason `def2_accumulator_ab` gives:
/// absolute times on this host vary up to 2x between processes, so only an
/// in-process A/B means anything. Both settings are one compiled program.
///
/// ```text
/// CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_2E_CHUNK_QUARTETS=256 \
///   cargo test --release -p cintx-oracle --features cpu,rocm,extended-device-rys,gth \
///   --test two_e_cooperative_arm -- --ignored --nocapture
/// ```
///
/// `CINTX_2E_CHUNK_QUARTETS` is not optional on a display GPU: an unbounded
/// dispatch here runs long enough to trip amdgpu's gfx job timeout, which
/// resets the device and takes the desktop session with it
/// (`gth_molopt_speed_memory_plan.md` §8.6).
#[cfg(feature = "rocm")]
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn split_g_build_beats_lane0_on_rocm() {
    if !std::env::var("CINTX_ROCM_ORACLE").is_ok_and(|value| value != "0") {
        println!("CINTX_ROCM_ORACLE not set; skipping");
        return;
    }
    let repeats = std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Rocm,
        ..Default::default()
    })
    .expect("rocm backend");

    let mut cases: Vec<(String, RawArrays)> = vec![(
        "H2O / def2-SVP".to_owned(),
        to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays"),
    )];
    #[cfg(feature = "gth")]
    cases.extend(
        def2_fixtures::gth_workloads()
            .into_iter()
            .filter(|(label, _)| label.starts_with("H2O")),
    );

    println!(
        "\nrocm — S3: split vs lane-0 G build, then the shared-memory slab \
         (best of {repeats}, interleaved)\n{:<30} {:>9} {:>7} {:>12} {:>12} {:>9} {:>13} {:>9}  {}",
        "case",
        "quartets",
        "nroots",
        "lane0 (ms)",
        "split (ms)",
        "speedup",
        "shared (ms)",
        "vs split",
        "identical"
    );
    for (label, arrays) in &cases {
        let shells = batch_shells(arrays);
        let list = full_list(arrays);
        let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
        let max_nroots = bucket_quartets(&view, &enumerate_quartets(&enumerate_pairs(&view)))
            .iter()
            .map(|bucket| bucket.class.nroots)
            .max()
            .unwrap_or(0);
        let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");

        // `g_in_shared` is comptime, so the shared-slab setting is a second
        // compiled program: warm both before timing either, or the first pass
        // measures the JIT rather than the kernel.
        for shared in [false, true] {
            cintx_cubecl::set_shared_g_enabled(shared);
            let _ = evaluate_2e_quartet_batch_resident(&backend, &resident, &list)
                .expect("rocm warm-up");
        }

        let mut lane0_ns = u64::MAX;
        let mut split_ns = u64::MAX;
        let mut shared_ns = u64::MAX;
        let mut lane0_values = Vec::new();
        let mut split_values = Vec::new();
        let mut shared_values = Vec::new();
        for _ in 0..repeats {
            for mode in 0..3u32 {
                set_cooperative_build_split(mode != 0);
                cintx_cubecl::set_shared_g_enabled(mode == 2);
                let start = std::time::Instant::now();
                let out = evaluate_2e_quartet_batch_resident(&backend, &resident, &list)
                    .expect("rocm 2e batch");
                let elapsed = start.elapsed().as_nanos() as u64;
                if mode == 0 {
                    lane0_ns = lane0_ns.min(elapsed);
                    lane0_values = out.values;
                } else if mode == 1 {
                    split_ns = split_ns.min(elapsed);
                    split_values = out.values;
                } else {
                    shared_ns = shared_ns.min(elapsed);
                    shared_values = out.values;
                }
            }
        }
        set_cooperative_build_split(true);
        cintx_cubecl::set_shared_g_enabled(false);

        let identical = lane0_values.len() == split_values.len()
            && lane0_values
                .iter()
                .zip(&split_values)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let shared_identical = shared_values.len() == split_values.len()
            && shared_values
                .iter()
                .zip(&split_values)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{:<30} {:>9} {:>7} {:>12.2} {:>12.2} {:>8.2}x {:>13.2} {:>8.2}x  {identical}/{shared_identical}",
            label,
            list.len(),
            max_nroots,
            lane0_ns as f64 / 1e6,
            split_ns as f64 / 1e6,
            lane0_ns as f64 / split_ns as f64,
            shared_ns as f64 / 1e6,
            split_ns as f64 / shared_ns as f64,
        );
        assert!(
            shared_identical,
            "{label}: the shared-memory G slab must not change a value"
        );
        assert!(
            identical,
            "{label}: the two G-build modes must agree bit for bit"
        );
    }
}
