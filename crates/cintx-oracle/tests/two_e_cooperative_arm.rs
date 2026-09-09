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
    set_two_e_cube_dim, set_two_e_kl_split, set_two_e_per_unit,
};
use cintx_driver::{BasisView, bucket_quartets, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, water};

/// Every test in this file pins process-global switches (`set_two_e_per_unit`,
/// `set_two_e_cube_dim`, `set_two_e_kl_split`), so two of them running on
/// the harness's parallel test threads would read each other's settings. One
/// lock, taken for the length of each test.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lanes per cube in the pinned cooperative runs.
///
/// Four, not the heuristic width: on the CubeCL CPU runtime the cube dimension
/// *is* the OS-thread count, and every `sync_cube` is a spin barrier across all
/// of them. Four is enough for the lane split to be real — with `nroots >= 2`
/// every lane owns at least one `(axis, root)` slice, and with `nroots == 1`
/// three of the four do — while staying cheap enough to run in CI.
const COOPERATIVE_LANES: u32 = 4;

/// Lanes per cube for the block gate (B1, plan §22): wide enough that a cube
/// builds several primitive quartets at once — at twelve lanes an
/// `nroots == 1` class blocks four and `nroots == 2` two — and no wider,
/// because on the CPU runtime every lane is an OS thread spinning at each of
/// the block's four barriers (24 lanes took five minutes).
const COOPERATIVE_BLOCK_LANES: u32 = 12;

/// The cooperative cube width [`evaluate_pinned`] pins; the block gate widens
/// it for its own run and restores it.
static PINNED_LANES: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(COOPERATIVE_LANES);

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
        Some(PINNED_LANES.load(std::sync::atomic::Ordering::Relaxed))
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
    let _serial = serial();
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
    let _serial = serial();
    for (label, arrays) in def2_fixtures::gth_workloads() {
        if !label.starts_with("H2O") {
            continue;
        }
        let list = one_quartet_per_class(&arrays, 3, 12);
        println!("\n{label}: {} quartets, one per class", list.len());
        assert_arms_agree(&label, &arrays, &list);
    }
}

/// B1 (plan §22): a cube wide enough to build a *block* of primitive quartets
/// at once — one per lane sub-group, into the shared tier's sub-slabs — and
/// then contract the block in row order. The claim is the same as S3's: every
/// element is still its own expression over the same operands, and the
/// accumulation order is the serial walk's, so the block is bit-identical to
/// the per-unit arm. The four-lane gates above never reach a block wider than
/// one; this one does, on the CPU runtime, where both arms can be pinned.
#[cfg(feature = "gth")]
#[test]
fn cooperative_block_is_bit_identical_on_gth() {
    let _serial = serial();
    PINNED_LANES.store(
        COOPERATIVE_BLOCK_LANES,
        std::sync::atomic::Ordering::Relaxed,
    );
    for (label, arrays) in def2_fixtures::gth_workloads() {
        if !label.starts_with("H2O") {
            continue;
        }
        let list = one_quartet_per_class(&arrays, 2, 6);
        println!(
            "\n{label}: {} quartets, one per class, {COOPERATIVE_BLOCK_LANES} lanes",
            list.len()
        );
        assert_arms_agree(&label, &arrays, &list);
    }
    PINNED_LANES.store(COOPERATIVE_LANES, std::sync::atomic::Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// G1 — the ket-pair split, exercised without a GPU
// ─────────────────────────────────────────────────────────────────────────────

/// The ket-pair split (G1) spreads a cooperative quartet over `n` cubes, each
/// accumulating a contiguous slice of the ket range into its own copy of the
/// output, and a reduce kernel sums the copies in part order. The default
/// never selects it on the CPU backend (no hardware planes), so it is pinned
/// here on the 4-lane cooperative arm and held to two things: the vendor at
/// the oracle tolerance, and the unsplit cooperative result to within a few
/// ULP of each block's scale — the sum over ket pairs is re-associated, so
/// bit-identity is not the gate, but a wrong row range or a partial written
/// on top of another is a wrong number, orders of magnitude outside it.
fn assert_split_agrees(label: &str, arrays: &RawArrays, list: &[[u32; 4]], parts: u32) {
    assert_split_agrees_on(label, arrays, list, parts, false);
}

/// [`assert_split_agrees`] on a named decomposition — `per_unit` is the CPU
/// shape, which takes the same split since §17 and where a part is a row of
/// K2's partition rather than a cube.
fn assert_split_agrees_on(
    label: &str,
    arrays: &RawArrays,
    list: &[[u32; 4]],
    parts: u32,
    per_unit: bool,
) {
    let arm = if per_unit { "per-unit" } else { "coop" };
    let label = &format!("{label} [{arm}]");
    let shells = batch_shells(arrays);
    set_two_e_kl_split(Some(1));
    let (unsplit, offsets) = evaluate_pinned(&shells, list, per_unit, true);
    set_two_e_kl_split(Some(parts));
    let (split, split_offsets) = evaluate_pinned(&shells, list, per_unit, true);
    set_two_e_kl_split(None);
    assert_eq!(offsets, split_offsets, "{label}: block layout");
    assert_eq!(unsplit.len(), split.len(), "{label}: output length");

    let total = unsplit.len();
    let mut worst_eps = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for (index, &start) in offsets.iter().enumerate() {
        let end = offsets.get(index + 1).copied().unwrap_or(total);
        let scale = unsplit[start..end]
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        for (a, b) in unsplit[start..end].iter().zip(&split[start..end]) {
            let abs = (a - b).abs();
            worst_abs = worst_abs.max(abs);
            if scale > 0.0 {
                worst_eps = worst_eps.max(abs / (scale * f64::EPSILON));
            }
        }
    }
    println!(
        "  {label:<34} split {parts:>2} vs unsplit: max|diff|={worst_abs:.3e} \
         ({worst_eps:.1} eps of block scale)"
    );
    // Re-associating a sum over up to 49 ket pairs, each itself a staged
    // contraction, moves a value by tens of ULP (77 eps measured on
    // TZVP-MOLOPT water at three parts); a topology fault — a wrong row range,
    // a partial written on top of another — moves it by orders of magnitude.
    // The vendor gate below is the real bound; this one only has to tell
    // those two apart.
    assert!(
        worst_eps <= 1024.0,
        "{label}: the ket-pair split differs from the unsplit cooperative arm by \
         {worst_eps:.1} eps of block scale (max |diff| {worst_abs:.3e})"
    );

    let vendor = vendor_values(arrays, list, &offsets, total);
    let mut over = 0_usize;
    let mut worst = 0.0_f64;
    for (v, a) in vendor.iter().zip(&split) {
        let diff = (v - a).abs();
        worst = worst.max(diff);
        if diff > 1e-12 {
            over += 1;
        }
    }
    assert_eq!(
        over, 0,
        "{label}: the ket-pair split has {over} elements over 1e-12 against vendored \
         libcint (max |diff| {worst:.3e})"
    );
    println!("  {label:<34} split {parts:>2} vs vendor:  max|diff|={worst:.3e}");
}

/// def2-SVP: segmented, short ket ranges — some parts of a wide split are
/// empty, which must contribute exactly zero.
#[test]
fn ket_split_agrees_on_def2() {
    let _serial = serial();
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let list = one_quartet_per_class(&arrays, 3, 24);
    for parts in [2, 7] {
        assert_split_agrees("H2O / def2-SVP", &arrays, &list, parts);
    }
}

/// GTH-MOLOPT: generally contracted, so every part runs the staged
/// contraction's stages and its own `out` flush, and the reduce sums those.
#[cfg(feature = "gth")]
#[test]
fn ket_split_agrees_on_gth() {
    let _serial = serial();
    for (label, arrays) in def2_fixtures::gth_workloads() {
        if !label.starts_with("H2O") {
            continue;
        }
        let list = one_quartet_per_class(&arrays, 3, 12);
        for parts in [3, 8] {
            assert_split_agrees(&label, &arrays, &list, parts);
        }
    }
}

/// The same split on the **per-unit** arm (§17).
///
/// It is the same host-side row expansion and the same reduce, but a different
/// kernel shape reads it — `lanes == 1`, no barrier, the staged contraction
/// running whole inside one unit — and a different consumer downstream: K2
/// partitions the *expanded* rows, so a per-row cost that still carried the
/// whole quartet's would make the partition believe the dispatch is `parts`
/// times more expensive than it is. Neither of those is exercised by the
/// cooperative cases above.
#[cfg(feature = "gth")]
#[test]
fn ket_split_agrees_on_gth_per_unit() {
    let _serial = serial();
    for (label, arrays) in def2_fixtures::gth_workloads() {
        if !label.starts_with("H2O") {
            continue;
        }
        let list = one_quartet_per_class(&arrays, 3, 12);
        for parts in [3, 8] {
            assert_split_agrees_on(&label, &arrays, &list, parts, true);
        }
    }
}

/// The same on def2-SVP's segmented, empty-part shape.
#[test]
fn ket_split_agrees_on_def2_per_unit() {
    let _serial = serial();
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let list = one_quartet_per_class(&arrays, 3, 24);
    for parts in [2, 7] {
        assert_split_agrees_on("H2O / def2-SVP", &arrays, &list, parts, true);
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
    let _serial = serial();
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
        for shared in [true, false] {
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
                // B1 (§22): the shared tier is the default; mode 2 is the
                // global-slab arm it replaced.
                cintx_cubecl::set_shared_g_enabled(mode != 2);
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
        cintx_cubecl::set_shared_g_enabled(true);

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
