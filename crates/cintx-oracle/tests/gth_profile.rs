//! GTH-MOLOPT profile: where the batched `int2e_sph` time and memory go.
//!
//! The development host has no hardware profiler for the CubeCL CPU runtime
//! (no `perf`, and the kernel is JIT-compiled MLIR), so attribution is done
//! the way `gth_molopt_speed_memory_plan.md` §9.3 does it: with the kernel's
//! runtime A/B switches, alternated inside one process, best of `N`. Every
//! number printed here is a ratio against the default configuration measured
//! in the same process; absolute times are not comparable across runs.
//!
//! `CINTX_GTH_BACKEND=rocm` runs the cooperative decomposition instead, with
//! `coop=lane0` (S3's A/B) in place of the per-unit variants; keep
//! `CINTX_2E_CHUNK_QUARTETS` set on a display GPU (plan §8.6).
//!
//! Variants (all one compiled program unless noted):
//!
//! - `default`      — per-unit decomposition at the heuristic width, staged
//!                    contraction, host transform.
//! - `units=1/4/8`  — `set_two_e_cube_dim`: the parallel-scaling curve. On a
//!                    16-thread host a `units=1 / default` ratio far below
//!                    the core count means load imbalance or contention, not
//!                    arithmetic (a different compiled program per width).
//! - `naive`        — `set_staged_contraction(false)`: the contraction arm.
//! - `balance=…`    — `set_two_e_balance`: uniform vs cost-balanced per-unit
//!                    partition (K2), when the kernel carries it.
//! - `xform=device` — `set_device_transform(Some(true))`: M3 on the CPU
//!                    runtime, for the memory column as much as the time.
//! - `fuse=off`     — `set_two_e_nroots_fusion(false)`: one dispatch per Rys
//!                    order, the grouping before F1 (§15). It is the one
//!                    variant that is a different *compiled program* rather
//!                    than a kernel scalar, because `nr_max` is comptime; it is
//!                    still interleaved in-process. On the per-unit arm both
//!                    arms must dump the same bits; on the cooperative arm they
//!                    need not, because the ket-pair split each chooses differs
//!                    (§16).
//! - `klsplit=off`  — `set_two_e_kl_split(Some(1))`: the whole ket range on one
//!                    work item, the shape before G1. It is also the arm that
//!                    restores bit-identity, on either decomposition, since the
//!                    split is the only thing in the 2e path that re-associates
//!                    a sum (§17).
//! - `coop=lane0`   — GPU only: the pre-S3 G build.
//!
//! Memory: the planned-bytes fields of `BatchExecutionStats` for every
//! variant, plus backend residency when `CINTX_BATCH_MEMORY_PROFILE=1`.
//!
//! `CINTX_GTH_DUMP=<dir>` writes each workload's **`klsplit=off`** output as raw
//! `f64`; `CINTX_GTH_COMPARE=<dir>` reads them back and holds that same arm to
//! them bit for bit. That is how a kernel change claiming bit-identity (K1's
//! index table, K2's partition, F1's grouping) is checked. The *default* arm
//! splits ket-pair ranges (§17) and so re-associates a sum by design; its
//! divergence from the reference is reported and bounded by the same
//! block-scale rule `two_e_cooperative_arm`'s forced-split gates use, rather
//! than asserted to be zero.
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle \
//!   --features cpu,extended-device-rys,gth --test gth_profile \
//!   -- --ignored --nocapture
//! ```

#![cfg(all(feature = "cpu", feature = "gth", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::RawArrays;
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    ResidentTwoEBasis, TwoEBatchOptions, TwoEBatchOutput, TwoEBatchStats as BatchExecutionStats,
    evaluate_2e_quartet_batch_resident, evaluate_2e_quartet_batch_with, prewarm_2e_work_list,
    set_contraction_probe, set_cooperative_build_split, set_device_transform,
    set_staged_contraction, set_two_e_balance, set_two_e_cube_dim, set_two_e_kl_split,
    set_two_e_nroots_fusion,
};
use cintx_driver::{BasisView, bucket_quartets, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, gth_workloads};

const VENDOR_TOLERANCE: f64 = 1e-12;

/// `CINTX_GTH_BACKEND=rocm` runs the same profile on the cooperative (GPU)
/// decomposition; the per-unit variants are skipped there.
fn backend_kind() -> BackendKind {
    match std::env::var("CINTX_GTH_BACKEND").as_deref() {
        #[cfg(feature = "rocm")]
        Ok("rocm") => BackendKind::Rocm,
        #[cfg(feature = "cuda")]
        Ok("cuda") => BackendKind::Cuda,
        Ok(other) if other != "cpu" => panic!("CINTX_GTH_BACKEND={other}: not compiled in"),
        _ => BackendKind::Cpu,
    }
}

fn backend() -> ResolvedBackend {
    let kind = backend_kind();
    let label = format!("{kind:?}");
    ResolvedBackend::from_intent(&BackendIntent {
        backend: kind,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{label} backend: {error}"))
}

fn repeats() -> usize {
    std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3)
}

fn quartet_list(arrays: &RawArrays) -> Vec<[u32; 4]> {
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect()
}

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

fn vendor_gap(vendor: &[f64], actual: &[f64]) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut over = 0_usize;
    for (v, a) in vendor.iter().zip(actual) {
        let diff = (v - a).abs();
        worst = worst.max(diff);
        if diff > VENDOR_TOLERANCE {
            over += 1;
        }
    }
    (worst, over)
}

/// One measured configuration.
struct Variant {
    name: &'static str,
    /// Apply the configuration; returns the residency to evaluate with.
    apply: fn(&ResolvedBackend, &[cintx_cubecl::BatchShell]) -> ResidentTwoEBasis,
    /// Evaluate under a `memory_limit_bytes` that caps the Cartesian
    /// intermediate at a quarter of the unchunked one (M1 chunking).
    limited: bool,
}

fn reset() {
    set_two_e_cube_dim(None);
    set_staged_contraction(true);
    set_two_e_balance(None);
    set_cooperative_build_split(true);
    set_two_e_kl_split(None);
    set_two_e_nroots_fusion(None);
    set_device_transform(Some(false));
}

fn resident(backend: &ResolvedBackend, shells: &[cintx_cubecl::BatchShell]) -> ResidentTwoEBasis {
    ResidentTwoEBasis::new(backend, shells).expect("residency")
}

fn variants() -> Vec<Variant> {
    let mut out = vec![Variant {
        name: "default",
        apply: |b, s| {
            reset();
            resident(b, s)
        },
        limited: false,
    }];
    let gpu = backend_kind() != BackendKind::Cpu;
    let scaling = std::env::var("CINTX_GTH_SCALING").is_ok_and(|v| v == "1") && !gpu;
    if scaling {
        out.push(Variant {
            name: "units=1",
            apply: |b, s| {
                reset();
                set_two_e_cube_dim(Some(1));
                resident(b, s)
            },
            limited: false,
        });
        out.push(Variant {
            name: "units=8",
            apply: |b, s| {
                reset();
                set_two_e_cube_dim(Some(8));
                resident(b, s)
            },
            limited: false,
        });
    }
    out.push(Variant {
        name: "naive",
        apply: |b, s| {
            reset();
            set_staged_contraction(false);
            resident(b, s)
        },
        limited: false,
    });
    out.push(Variant {
        name: "fuse=off",
        apply: |b, s| {
            reset();
            set_two_e_nroots_fusion(Some(false));
            resident(b, s)
        },
        limited: false,
    });
    out.push(Variant {
        name: "klsplit=off",
        apply: |b, s| {
            reset();
            set_two_e_kl_split(Some(1));
            resident(b, s)
        },
        limited: false,
    });
    out.push(Variant {
        name: "probe:no-ctr",
        apply: |b, s| {
            reset();
            set_contraction_probe();
            resident(b, s)
        },
        limited: false,
    });
    if gpu {
        out.push(Variant {
            name: "coop=lane0",
            apply: |b, s| {
                reset();
                set_cooperative_build_split(false);
                resident(b, s)
            },
            limited: false,
        });
    } else {
        out.push(Variant {
            name: "balance=uniform",
            apply: |b, s| {
                reset();
                set_two_e_balance(Some(false));
                resident(b, s)
            },
            limited: false,
        });
    }
    out.push(Variant {
        name: "xform=device",
        apply: |b, s| {
            reset();
            set_device_transform(Some(true));
            resident(b, s)
        },
        limited: false,
    });
    out.push(Variant {
        name: "chunk=cart/4",
        apply: |b, s| {
            reset();
            resident(b, s)
        },
        limited: true,
    });
    out
}

struct Measured {
    name: &'static str,
    best_ns: u64,
    stats: BatchExecutionStats,
    values: Vec<f64>,
    offsets: Vec<usize>,
}

fn sanitize(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn class_census(arrays: &RawArrays, list_len: usize) {
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&view));
    let buckets = bucket_quartets(&view, &quartets);
    println!(
        "  {} quartets in {} l-classes; per class: (l) nroots quartets prim-quartets",
        list_len,
        buckets.len()
    );
    let mut rows: Vec<(u32, [u8; 4], usize, u64)> = buckets
        .iter()
        .map(|bucket| {
            let prim: u64 = bucket
                .quartets
                .iter()
                .map(|&q| cintx_driver::primitive_work(&view, q))
                .sum();
            (
                bucket.class.nroots,
                bucket.class.angular_momenta,
                bucket.len(),
                prim,
            )
        })
        .collect();
    rows.sort_by_key(|r| (r.0, r.1));
    let total_prim: u64 = rows.iter().map(|r| r.3).sum();
    for (nroots, l, n, prim) in rows {
        println!(
            "    ({},{},{},{})  nroots={nroots}  quartets={n:>6}  prim={prim:>10} ({:.1}%)",
            l[0],
            l[1],
            l[2],
            l[3],
            100.0 * prim as f64 / total_prim as f64
        );
    }
}

fn run_workload(label: &str, arrays: &RawArrays) -> Vec<Measured> {
    let shells = batch_shells(arrays);
    let list = quartet_list(arrays);
    let backend = backend();
    println!("\n== {label} ==");
    class_census(arrays, list.len());

    // `memory_limit_bytes` for the chunked variant: the caller's output plus
    // the Cartesian intermediate, which `chunk_cart_budget` turns into a
    // Cartesian ceiling of half the unchunked one.
    //
    // It was half the intermediate until §18. The ket-pair split is a pure
    // function of the quartet now, so it runs under a budget too — a budgeted
    // run and an unbudgeted one have to compute the same values — and its
    // partial blocks are charged to the same ledger. That raises the *floor* a
    // budget can reach by roughly `(parts - 1)` copies of the widest Cartesian
    // block, and the old figure sat below the new floor for H2O/DZVP-MOLOPT-SR:
    // 679 KiB needed against 648 KiB allowed, with the chunk planner already
    // down to one block per chunk. A caller who needs the older, lower floor
    // turns the split off (`CINTX_2E_KL_SPLIT=off`) and accepts the unsplit
    // arithmetic; that is an explicit choice rather than one a memory limit
    // makes silently.
    let (output_bytes, cart_bytes) = list.iter().fold((0_usize, 0_usize), |(o, c), q| {
        let (mut sph, mut cart) = (1_usize, 1_usize);
        for &s in q {
            let shell = &shells[s as usize];
            sph *= cintx_cubecl::transform::c2s::nsph(shell.l) * shell.nctr as usize;
            cart *= cintx_cubecl::transform::c2s::ncart(shell.l) * shell.nctr as usize;
        }
        (o + sph * 8, c + cart * 8)
    });
    let limit = output_bytes + cart_bytes;

    let variants = variants();
    let residents: Vec<ResidentTwoEBasis> = variants
        .iter()
        .map(|v| (v.apply)(&backend, &shells))
        .collect();
    // One prewarm per distinct compiled program: the width variants compile
    // their own; everything else shares the default's.
    for v in &variants {
        (v.apply)(&backend, &shells);
        prewarm_2e_work_list(&backend, &shells, &list).expect("prewarm");
    }

    let mut best = vec![u64::MAX; variants.len()];
    let mut last: Vec<Option<TwoEBatchOutput>> = (0..variants.len()).map(|_| None).collect();
    for _ in 0..repeats() {
        for (index, v) in variants.iter().enumerate() {
            (v.apply)(&backend, &shells);
            let start = std::time::Instant::now();
            let out = if v.limited {
                evaluate_2e_quartet_batch_with(
                    &backend,
                    &residents[index],
                    &list,
                    TwoEBatchOptions {
                        memory_limit_bytes: Some(limit),
                        ..Default::default()
                    },
                )
                .expect(v.name)
            } else {
                evaluate_2e_quartet_batch_resident(&backend, &residents[index], &list)
                    .expect(v.name)
            };
            best[index] = best[index].min(start.elapsed().as_nanos() as u64);
            last[index] = Some(out);
        }
    }
    reset();

    let measured: Vec<Measured> = variants
        .iter()
        .zip(best)
        .zip(last)
        .map(|((v, best_ns), out)| {
            let out = out.expect("at least one repeat");
            Measured {
                name: v.name,
                best_ns,
                stats: out.stats,
                values: out.values,
                offsets: out.offsets,
            }
        })
        .collect();

    let base = &measured[0];
    let vendor = vendor_values(arrays, &list, &base.offsets, base.values.len());
    println!(
        "  {:<16} {:>10} {:>8}  {:>9} {:>9} {:>9} {:>9} {:>10}  {:>10}",
        "variant",
        "best (ms)",
        "ratio",
        "out MiB",
        "cart MiB",
        "rdbk MiB",
        "gslab KiB",
        "chunks",
        "vendor|d|"
    );
    for m in &measured {
        if m.name.starts_with("probe:") {
            println!(
                "  {:<16} {:>10.2} {:>7.3}x  (G build only; output undefined)",
                m.name,
                m.best_ns as f64 / 1e6,
                base.best_ns as f64 / m.best_ns as f64,
            );
            continue;
        }
        let (worst, over) = vendor_gap(&vendor, &m.values);
        let identical = m
            .values
            .iter()
            .zip(&base.values)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "  {:<16} {:>10.2} {:>7.3}x  {:>9.2} {:>9.2} {:>9.2} {:>9.1} {:>10}  {:>10.2e}{}{}",
            m.name,
            m.best_ns as f64 / 1e6,
            base.best_ns as f64 / m.best_ns as f64,
            m.stats.host_output_bytes as f64 / 1048576.0,
            m.stats.host_cart_bytes_peak as f64 / 1048576.0,
            m.stats.readback_bytes as f64 / 1048576.0,
            m.stats.device_g_slab_bytes_peak as f64 / 1024.0,
            m.stats.chunk_count,
            worst,
            if over > 0 { " OVER" } else { "" },
            if identical { " =bits" } else { "" },
        );
    }
    println!(
        "  launches={} classes={} chunks={} kl_split={} prim evaluated/total={}/{} dispatch={:.1}ms transform={:.1}ms",
        base.stats.kernel_launch_count,
        base.stats.launch_classes,
        base.stats.chunk_count,
        base.stats.kl_split_max,
        base.stats.primitive_quartets_evaluated,
        base.stats.primitive_quartets_total,
        base.stats.dispatch_ns as f64 / 1e6,
        base.stats.host_transform_ns as f64 / 1e6,
    );

    let (_, over) = vendor_gap(&vendor, &base.values);
    assert_eq!(
        over, 0,
        "{label}: default variant disagrees with the vendor"
    );

    // The launch floor: the cheapest quartet of every l-class, so every
    // dispatch of the full list is launched with almost no arithmetic behind
    // it. What is left is thread wake-up, tables and readback per launch.
    {
        let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
        let quartets = enumerate_quartets(&enumerate_pairs(&view));
        let floor_list: Vec<[u32; 4]> = bucket_quartets(&view, &quartets)
            .iter()
            .filter_map(|bucket| bucket.quartets.first())
            .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
            .collect();
        reset();
        let mut floor_ns = u64::MAX;
        let mut floor_stats = BatchExecutionStats::default();
        for _ in 0..repeats() {
            let start = std::time::Instant::now();
            let out = evaluate_2e_quartet_batch_resident(&backend, &residents[0], &floor_list)
                .expect("floor");
            floor_ns = floor_ns.min(start.elapsed().as_nanos() as u64);
            floor_stats = out.stats;
        }
        println!(
            "  launch floor: {} quartets (one per class) in {} launches: {:.2} ms = {:.1}% of default",
            floor_list.len(),
            floor_stats.kernel_launch_count,
            floor_ns as f64 / 1e6,
            100.0 * floor_ns as f64 / base.best_ns as f64
        );
    }

    // The dump/compare pair works on the **unsplit** arm, not the default one.
    //
    // Since §17 the default splits ket-pair ranges on both decompositions, and
    // that re-associates the sum over ket pairs by design — so the default's
    // bits are not stable across a change to the split, and asserting on them
    // would make this tool fail for the one reason it is not looking for.
    // `klsplit=off` is the arm whose bits *are* claimed stable, and holding it
    // to them keeps the strict rule exactly where it still applies (K1's index
    // table, K2's partition, F1's grouping). The default is still compared, and
    // reported, against the same block-scale bound the forced-split gates use.
    let strict = measured
        .iter()
        .find(|m| m.name == "klsplit=off")
        .unwrap_or(base);

    if let Ok(dir) = std::env::var("CINTX_GTH_DUMP") {
        let path = std::path::Path::new(&dir).join(format!("{}.f64", sanitize(label)));
        let bytes: Vec<u8> = strict.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("dump");
        println!(
            "  dumped {} unsplit values to {}",
            strict.values.len(),
            path.display()
        );
    }
    if let Ok(dir) = std::env::var("CINTX_GTH_COMPARE") {
        let path = std::path::Path::new(&dir).join(format!("{}.f64", sanitize(label)));
        let bytes = std::fs::read(&path).expect("reference dump");
        let reference: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(reference.len(), base.values.len(), "{label}: dump length");
        let differing = reference
            .iter()
            .zip(&strict.values)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = reference
            .iter()
            .zip(&strict.values)
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        // The default's divergence is reported, not asserted here: it is the
        // ket-pair split's re-association, which `ket_split_agrees_on_*` bounds
        // and the vendor column above gates.
        let split_eps = block_scale_eps(&reference, &base.values, &base.offsets);
        println!(
            "  vs {}: unsplit {differing} of {} elements differ (max|d|={worst:.3e}); \
             default {split_eps:.1} eps of block scale",
            path.display(),
            reference.len()
        );
        assert_eq!(
            differing, 0,
            "{label}: the unsplit arm is not bit-identical to the reference dump"
        );
        assert!(
            split_eps <= 1024.0,
            "{label}: the split default is {split_eps:.1} eps of block scale from the \
             reference dump, which is past what re-associating the ket-pair sum explains"
        );
    }
    measured
}

/// The worst per-block relative deviation between two runs, in ULP of the
/// block's own scale — the measure `two_e_cooperative_arm` bounds the ket-pair
/// split by, so the two gates speak the same units.
fn block_scale_eps(reference: &[f64], actual: &[f64], offsets: &[usize]) -> f64 {
    let total = reference.len();
    let mut worst = 0.0_f64;
    for (index, &start) in offsets.iter().enumerate() {
        let end = offsets.get(index + 1).copied().unwrap_or(total);
        let scale = reference[start..end]
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if scale <= 0.0 {
            continue;
        }
        for (a, b) in reference[start..end].iter().zip(&actual[start..end]) {
            worst = worst.max((a - b).abs() / (scale * f64::EPSILON));
        }
    }
    worst
}

fn workloads() -> Vec<(String, RawArrays)> {
    let filter = std::env::var("CINTX_GTH_FILTER").ok();
    let full = std::env::var("CINTX_BENCH_SCOPE").as_deref() == Ok("full");
    gth_workloads()
        .into_iter()
        .filter(|(label, _)| full || !label.starts_with("C6H6"))
        .filter(|(label, _)| filter.as_deref().is_none_or(|f| label.contains(f)))
        .collect()
}

#[test]
#[ignore = "profile; run explicitly in release with --ignored --nocapture"]
fn gth_profile() {
    for (label, arrays) in workloads() {
        run_workload(&label, &arrays);
    }
}
