//! Phase 36 — whole-workload throughput benchmark: cintx vs vendored libcint.
//!
//! The existing `benchmark_speed.rs` measures **one shell tuple at a time**,
//! which is the shape that makes CubeCL look 194x slower than libcint: each
//! call pays a full kernel launch, a fresh coefficient upload, and a blocking
//! readback. That per-tuple number is real, but it is not the number that
//! decides whether a production workload is faster.
//!
//! This benchmark measures the unit of work an SCF iteration actually needs:
//! the **entire screened shell-quartet list** of a molecule in a real basis
//! (def2-SVP / def2-TZVP), driven through `cintx-driver`.
//!
//! # Benchmark honesty rules (enforced here, not just documented)
//!
//! 1. **Same work on both sides.** Both engines run the identical screened
//!    work-list produced by the same [`SchwarzTable`]. The table is built once,
//!    from the reference engine, so screening cannot advantage either side.
//! 2. **Screened and unscreened are reported separately**, because screening is
//!    an algorithmic win and attributing it to a kernel would be dishonest.
//! 3. **Results are compared, not just timed.** A speed number is only printed
//!    for a configuration whose values match the reference.
//! 4. **Coverage is reported.** Quartets cintx cannot evaluate are counted and
//!    printed rather than silently skipped.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_throughput_benchmark -- --ignored --nocapture`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_driver::{
    BasisView, DiagonalEvaluator, DriverError, LaunchTier, QuartetEvaluator, ShellPair,
    ShellQuartet, bucket_quartets, build_schwarz_table, enumerate_pairs, enumerate_quartets,
    run_buckets, screen_quartets,
};
use cintx_oracle::vendor_ffi;
use def2_fixtures::{methane, sulfur_dioxide, water};
use serde_json::{Value, json};
use std::sync::Mutex;

/// Shared-memory budget used only for the tier histogram, not for dispatch.
const SHARED_MEMORY_BYTES: usize = 48 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Engine adapters
// ─────────────────────────────────────────────────────────────────────────────

struct VendorEngine<'a> {
    arrays: &'a RawArrays,
}

impl VendorEngine<'_> {
    fn eval(&self, shls: &[i32; 4], out: &mut [f64]) -> i32 {
        vendor_ffi::vendor_int2e_sph(
            out,
            shls,
            &self.arrays.atm,
            self.arrays.natm() as i32,
            &self.arrays.bas,
            self.arrays.nbas() as i32,
            &self.arrays.env,
        )
    }
}

impl QuartetEvaluator for VendorEngine<'_> {
    fn eval_quartet(&mut self, quartet: ShellQuartet, out: &mut [f64]) -> Result<(), DriverError> {
        self.eval(&quartet.shls(), out);
        Ok(())
    }
    fn engine_name(&self) -> &'static str {
        "libcint 6.1.3 (C, 1 thread)"
    }
}

impl DiagonalEvaluator for VendorEngine<'_> {
    fn eval_diagonal(&mut self, pair: ShellPair, out: &mut [f64]) -> Result<(), DriverError> {
        self.eval(
            &[pair.i as i32, pair.j as i32, pair.i as i32, pair.j as i32],
            out,
        );
        Ok(())
    }
}

struct CintxEngine<'a> {
    arrays: &'a RawArrays,
}

impl QuartetEvaluator for CintxEngine<'_> {
    fn eval_quartet(&mut self, quartet: ShellQuartet, out: &mut [f64]) -> Result<(), DriverError> {
        let shls = quartet.shls();
        // SAFETY: `out` is sized by the driver from the same `bas` array that
        // `eval_raw` reads its extents from.
        unsafe {
            eval_raw(
                RawApiId::INT2E_SPH,
                Some(out),
                None,
                &shls,
                &self.arrays.atm,
                &self.arrays.bas,
                &self.arrays.env,
                None,
                None,
            )
        }
        .map(|_| ())
        .map_err(|error| DriverError::Evaluation {
            shells: shls,
            detail: error.to_string(),
        })
    }
    fn engine_name(&self) -> &'static str {
        "cintx CubeCL (cpu backend)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The machine-readable record (D0.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Rows collected by every `run_case` / `run_batch_case` in this process.
///
/// A benchmark that only prints cannot be compared against last week's, and the
/// plan's G5 says a speed number is only admissible with the work list, the
/// coverage and the match status that produced it. So every row a run produces
/// is accumulated here and written once, rather than each case writing its own
/// file and the last one winning.
static ROWS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

fn record(row: Value) {
    ROWS.lock().expect("bench row sink poisoned").push(row);
}

/// Per-bucket census of one work list: class, Rys order, tier, quartet count.
///
/// This is D0.3's baseline field. "def2-TZVP has classes above nroots 5" is a
/// yes/no; what makes D1's progress measurable is *how much work* sits in them,
/// and that is a per-bucket number that has to be recorded before the coverage
/// changes underneath it.
fn bucket_rows(basis: &BasisView<'_>, buckets: &[cintx_driver::Bucket]) -> Vec<Value> {
    buckets
        .iter()
        .map(|bucket| {
            let tier = bucket.class.tier(SHARED_MEMORY_BYTES);
            json!({
                "class": bucket.class.angular_momenta,
                "nroots": bucket.class.nroots,
                "quartets": bucket.len(),
                "g_tensor_bytes": bucket.class.g_tensor_bytes(),
                "tier": format!("{tier:?}"),
                "above_base_ceiling": bucket.class.nroots as usize
                    > cintx_cubecl::BASE_DEVICE_NROOTS,
                // The contraction shape D2.4 wants per bucket: a class that is
                // contraction-bound rather than angular-momentum-bound is the
                // one that should prefer the cooperative kernel arm.
                "primitive_work": bucket
                    .quartets
                    .iter()
                    .map(|&q| cintx_driver::primitive_work(basis, q))
                    .sum::<u64>(),
            })
        })
        .collect()
}

/// Split of a bucket list by the base device envelope — the D0.3 baseline.
fn envelope_split(buckets: &[cintx_driver::Bucket]) -> Value {
    let (mut above, mut below) = (0_usize, 0_usize);
    for bucket in buckets {
        if bucket.class.nroots as usize > cintx_cubecl::BASE_DEVICE_NROOTS {
            above += bucket.len();
        } else {
            below += bucket.len();
        }
    }
    json!({
        "quartets_at_or_below_base_ceiling": below,
        "quartets_above_base_ceiling": above,
        "fraction_above_base_ceiling": if above + below == 0 {
            0.0
        } else {
            above as f64 / (above + below) as f64
        },
    })
}

/// Write everything this process collected to the mandatory artifact locations.
///
/// Called from each `#[test]` entry point after its cases run. Writing on every
/// call rather than once at exit means a run interrupted halfway still leaves
/// the rows it did produce, which is the difference between a partial record and
/// none.
fn flush_artifact(run: &str) {
    let rows = ROWS.lock().expect("bench row sink poisoned").clone();
    if rows.is_empty() {
        return;
    }
    let artifact = json!({
        "schema": "cintx_def2_throughput/1",
        "run": run,
        "backend": "cpu",
        "reference_engine": "libcint 6.1.3 (C, 1 thread)",
        "extended_device_rys": cfg!(feature = "extended-device-rys"),
        "base_nroots_ceiling": cintx_cubecl::BASE_DEVICE_NROOTS,
        "quartet_cap": quartet_cap(),
        "repeats": bench_repeats(),
        "cases": rows,
    });
    match cintx_oracle::fixtures::write_pretty_json_artifact(
        "/mnt/data/cintx_def2_throughput.json",
        "cintx_def2_throughput.json",
        &artifact,
    ) {
        Ok(written) => println!("\nthroughput artifact: {}", written.actual_path.display()),
        Err(error) => eprintln!("\nthroughput artifact NOT written: {error}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The benchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Cap on quartets evaluated per case.
///
/// The full def2-SVP water list is ~3.1 k quartets; at the current per-quartet
/// cost that takes over half an hour, so the benchmark samples a bounded,
/// bucket-proportional prefix instead. The sample is taken **after** bucketing
/// and applied identically to both engines, so the ratio is unaffected; only
/// the absolute wall-clock is a sample.
fn quartet_cap() -> usize {
    std::env::var("CINTX_BENCH_CAP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(240)
}

/// How many timed repeats each engine gets in the batched benchmark.
///
/// One shot is not a measurement: the whole H2O/def2-SVP work list runs in a
/// few milliseconds, which is well inside the noise of scheduler wake-ups and
/// cache state. Both engines get the same number of repeats and both are
/// reported by their **best** run, which is the standard estimator for a
/// deterministic workload — the minimum is the sample least polluted by
/// unrelated system activity, and using it for both sides keeps the ratio fair.
fn bench_repeats() -> usize {
    std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(9)
}

fn run_case(label: &str, molecule: &Molecule, tolerance: f64) {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);

    let pairs = enumerate_pairs(&basis);
    let quartets = enumerate_quartets(&pairs);

    let mut vendor = VendorEngine { arrays: &arrays };
    let table = build_schwarz_table(&basis, &pairs, &mut vendor).expect("schwarz table");
    let (kept, report) = screen_quartets(&quartets, &table, tolerance);
    let mut buckets = bucket_quartets(&basis, &kept);

    // Sample a bounded, bucket-proportional prefix so the run terminates.
    // Every launch class keeps at least one quartet, so the class histogram
    // (and its warm-up cost) stays representative.
    let cap = quartet_cap();
    let total_kept: usize = buckets.iter().map(|b| b.len()).sum();
    if total_kept > cap {
        for bucket in &mut buckets {
            let take = ((bucket.len() * cap) / total_kept).max(1);
            bucket.quartets.truncate(take);
        }
    }
    let sampled: usize = buckets.iter().map(|b| b.len()).sum();

    println!("\n{}", "=".repeat(100));
    println!("{label}");
    println!("{}", "=".repeat(100));
    println!(
        "  shells={}  pairs={}  quartets(8-fold)={}  kept={} ({:.1}%)  tol={:.0e}  buckets={}",
        basis.nbas(),
        pairs.len(),
        quartets.len(),
        report.kept,
        report.kept_fraction() * 100.0,
        tolerance,
        buckets.len()
    );
    if sampled < total_kept {
        println!(
            "  SAMPLED {sampled} of {total_kept} kept quartets (cap={cap}); wall-clock \
             below is for the sample -- the cintx/libcint ratio is unaffected."
        );
    }

    // Reference run.
    let mut vendor = VendorEngine { arrays: &arrays };
    let reference =
        run_buckets(&basis, &buckets, &mut vendor, SHARED_MEMORY_BYTES).expect("vendor batch run");

    // cintx warm-up: the CubeCL backend specializes (and, on the CPU runtime,
    // JIT-compiles) a kernel per distinct shape. That cost is paid once per
    // launch class, not per integral, so it must be measured separately or it
    // gets attributed to throughput it does not belong to.
    let mut cintx = CintxEngine { arrays: &arrays };
    let warmup_start = std::time::Instant::now();
    let mut warm_scratch = vec![0.0_f64; 4096];
    for bucket in &buckets {
        if let Some(&quartet) = bucket.quartets.first() {
            let len = cintx_driver::execute::block_len(&basis, quartet);
            if warm_scratch.len() < len {
                warm_scratch.resize(len, 0.0);
            }
            let _ = cintx.eval_quartet(quartet, &mut warm_scratch[..len]);
        }
    }
    let warmup = warmup_start.elapsed();
    println!(
        "  cintx warm-up (one quartet per launch class, {} classes): {:.3} s  [{:.1} ms/class]",
        buckets.len(),
        warmup.as_secs_f64(),
        warmup.as_secs_f64() * 1000.0 / buckets.len().max(1) as f64
    );

    // cintx steady-state run over the identical work-list.
    let actual =
        run_buckets(&basis, &buckets, &mut cintx, SHARED_MEMORY_BYTES).expect("cintx batch run");

    let tiers = reference.stats.tier_counts;
    println!(
        "  launch tiers: thread-per-quartet={}  cube+shared={}  cube+global={}",
        tiers[0], tiers[1], tiers[2]
    );
    // The envelope report, against the ceiling this build actually has. Printing
    // a literal `nroots<=5` here would have gone stale the moment D1 raised the
    // ceiling, and would have read as a coverage gap where there is none.
    let device_ceiling = cintx_cubecl::device_nroots_ceiling(
        &cintx_cubecl::backend::ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend"),
        cintx_cubecl::RysFamily::Int2e,
    );
    let eligible: usize = buckets
        .iter()
        .filter(|b| b.class.nroots as usize <= device_ceiling)
        .map(|b| b.len())
        .sum();
    let above_base: usize = buckets
        .iter()
        .filter(|b| b.class.nroots as usize > cintx_cubecl::BASE_DEVICE_NROOTS)
        .map(|b| b.len())
        .sum();
    println!(
        "  device-eligible (nroots<={device_ceiling}): {eligible} of {} sampled quartets  \
         [{above_base} of them above the base ceiling {}]",
        buckets.iter().map(|b| b.len()).sum::<usize>(),
        cintx_cubecl::BASE_DEVICE_NROOTS,
    );

    // Correctness before speed: only compare timings if values agree.
    let mut max_diff = 0.0_f64;
    let mut mismatches = 0_usize;
    for (index, (&expected_offset, quartet)) in reference
        .offsets
        .iter()
        .zip(&reference.quartets)
        .enumerate()
    {
        let len = cintx_driver::execute::block_len(&basis, *quartet);
        let e = &reference.values[expected_offset..expected_offset + len];
        let a = &actual.values[actual.offsets[index]..actual.offsets[index] + len];
        for (x, y) in e.iter().zip(a) {
            let diff = (x - y).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-9 {
                mismatches += 1;
            }
        }
    }

    println!(
        "  cintx coverage: evaluated={} failed={}  max|diff| vs vendor={:.3e}  mismatched elements={}",
        actual.stats.quartets_evaluated, actual.stats.quartets_failed, max_diff, mismatches
    );

    let ref_secs = reference.stats.elapsed.as_secs_f64();
    let act_secs = actual.stats.elapsed.as_secs_f64();
    println!(
        "\n  {:<32} {:>12} {:>16} {:>16}",
        "engine", "wall (s)", "quartets/s", "integrals/s"
    );
    for (name, stats) in [
        ("libcint 6.1.3 (C, 1 thread)", &reference.stats),
        ("cintx CubeCL (cpu backend)", &actual.stats),
    ] {
        let secs = stats.elapsed.as_secs_f64();
        println!(
            "  {:<32} {:>12.4} {:>16.1} {:>16.3e}",
            name,
            secs,
            stats.quartets_evaluated as f64 / secs.max(f64::MIN_POSITIVE),
            stats.integrals_per_second().unwrap_or(0.0)
        );
    }

    if actual.stats.quartets_failed == 0 && mismatches == 0 {
        println!(
            "\n  VERDICT: cintx is {:.2}x {} than libcint on this workload.",
            if act_secs > ref_secs {
                act_secs / ref_secs
            } else {
                ref_secs / act_secs
            },
            if act_secs > ref_secs {
                "SLOWER"
            } else {
                "FASTER"
            }
        );
    } else {
        println!(
            "\n  VERDICT: NOT COMPARABLE — {} quartets failed, {} elements mismatched. \
             Speed is not reported for an incorrect or incomplete run.",
            actual.stats.quartets_failed, mismatches
        );
    }

    record(json!({
        "case": label,
        "mode": "per-quartet",
        "tolerance": tolerance,
        "shells": basis.nbas(),
        "pairs": pairs.len(),
        "quartets_enumerated": quartets.len(),
        "quartets_kept": report.kept,
        "kept_fraction": report.kept_fraction(),
        "quartets_sampled": sampled,
        "buckets": buckets.len(),
        "envelope": envelope_split(&buckets),
        "bucket_rows": bucket_rows(&basis, &buckets),
        "tier_counts": {
            "thread_per_quartet": tiers[0],
            "cube_per_quartet_shared": tiers[1],
            "cube_per_quartet_global": tiers[2],
        },
        "warmup_seconds": warmup.as_secs_f64(),
        "libcint_seconds": ref_secs,
        "cintx_seconds": act_secs,
        "quartets_evaluated": actual.stats.quartets_evaluated,
        "quartets_failed": actual.stats.quartets_failed,
        "max_abs_diff_vs_vendor": max_diff,
        "mismatched_elements": mismatches,
        "comparable": actual.stats.quartets_failed == 0 && mismatches == 0,
    }));
}

#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_whole_workload_throughput() {
    println!("\nCPU-backend note: CubeCL and libcint run on the same silicon here, so this");
    println!("case measures the per-quartet route, where cintx pays a planner pass, twelve");
    println!("allocations, a dispatch and a blocking readback per shell quartet. It is the");
    println!("cost `def2_batched_throughput` removes, and it is reported so the gap between");
    println!("the two routes stays visible rather than being quoted away.\n");

    // Scope is opt-in: the heavier fixtures multiply the wall clock of a route
    // that is already the slow one, so they are gated behind
    // CINTX_BENCH_SCOPE=full.
    let scope = std::env::var("CINTX_BENCH_SCOPE").unwrap_or_else(|_| "svp".to_owned());

    run_case(
        "H2O / def2-SVP  (unscreened)",
        &water(StandardBasis::Def2Svp),
        0.0,
    );
    run_case(
        "H2O / def2-SVP  (screened 1e-10)",
        &water(StandardBasis::Def2Svp),
        1e-10,
    );

    if scope == "full" {
        run_case(
            "CH4 / def2-SVP  (screened 1e-10)",
            &methane(StandardBasis::Def2Svp),
            1e-10,
        );
        run_case(
            "SO2 / def2-SVP  (screened 1e-10)",
            &sulfur_dioxide(StandardBasis::Def2Svp),
            1e-10,
        );
        run_case(
            "H2O / def2-TZVP (screened 1e-10)",
            &water(StandardBasis::Def2Tzvp),
            1e-10,
        );
        // The second-row TZVP case (D0.1). It is last because it is the
        // heaviest: SO2/def2-TZVP is 35 shells, and it is the only fixture
        // whose nroots 6-7 buckets carry enough quartets to weigh anything in a
        // TZVP timing.
        run_case(
            "SO2 / def2-TZVP (screened 1e-10)",
            &sulfur_dioxide(StandardBasis::Def2Tzvp),
            1e-10,
        );
    } else {
        println!(
            "\n(CH4, SO2 and def2-TZVP cases skipped; set CINTX_BENCH_SCOPE=full to \
             include them.)"
        );
    }

    flush_artifact("def2_whole_workload_throughput");
}

/// The screening correctness gate: at tolerance 0 the kept list must be the
/// full list, so screening can never change results — only cost.
#[test]
fn zero_tolerance_screening_is_the_identity() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).unwrap();
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let pairs = enumerate_pairs(&basis);
    let quartets = enumerate_quartets(&pairs);

    let mut vendor = VendorEngine { arrays: &arrays };
    let table = build_schwarz_table(&basis, &pairs, &mut vendor).unwrap();

    let (kept, report) = screen_quartets(&quartets, &table, 0.0);
    assert_eq!(kept.len(), quartets.len());
    assert_eq!(report.skipped(), 0);

    // And the screen must not be a silent no-op. The bound is
    // `|(ij|kl)| <= Q_ij * Q_kl`, so a tolerance just under `max_q^2` can keep
    // at most the quartets built from the very largest pairs and must drop the
    // rest. Deriving the tolerance from the table rather than hardcoding one
    // keeps this a real test of the screen instead of a test of how diffuse
    // this particular molecule happens to be.
    //
    // H2O/def2-SVP is deliberately NOT asserted to lose work at a production
    // tolerance: every shell pair here sits within ~1.8 bohr, so every
    // `Q_ij * Q_kl` is far above 1e-10 and keeping all 3081 quartets is the
    // correct answer, not a screening failure.
    let aggressive = table.max() * table.max() * 0.999;
    let (screened, screened_report) = screen_quartets(&quartets, &table, aggressive);
    assert!(
        screened.len() < quartets.len(),
        "screening at tolerance {aggressive:e} (max_q^2 = {:e}) removed nothing: \
         {screened_report:?}",
        table.max() * table.max()
    );

    // A tolerance above the largest possible product must remove everything.
    let (none, _) = screen_quartets(&quartets, &table, table.max() * table.max() * 1.001);
    assert!(
        none.is_empty(),
        "no quartet can exceed max_q^2, so all must be screened; kept {}",
        none.len()
    );
}

/// Every quartet the driver produces for def2-SVP must be device-eligible
/// (nroots<=5) and fit the shared-memory tier — the plan's claim that def2-SVP
/// lands entirely inside the existing device envelope.
#[test]
fn def2_svp_fits_the_device_envelope() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).unwrap();
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));

    for bucket in bucket_quartets(&basis, &quartets) {
        assert!(
            bucket.class.nroots <= 5,
            "def2-SVP class {:?} needs nroots={} (>5)",
            bucket.class.angular_momenta,
            bucket.class.nroots
        );
        assert_ne!(
            bucket.class.tier(SHARED_MEMORY_BYTES),
            LaunchTier::CubePerQuartetGlobal,
            "def2-SVP class {:?} should fit shared memory",
            bucket.class.angular_momenta
        );
    }
}

/// def2-TZVP must contain classes that exceed the device envelope — the
/// concrete reason Phase 33 (device Rys nroots 6..12) is required for TZVP.
#[test]
fn def2_tzvp_exceeds_the_device_envelope() {
    let molecule = water(StandardBasis::Def2Tzvp);
    let arrays = to_raw_arrays(&molecule).unwrap();
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    let buckets = bucket_quartets(&basis, &quartets);

    let over: usize = buckets
        .iter()
        .filter(|b| b.class.nroots > 5)
        .map(|b| b.len())
        .sum();
    let total: usize = buckets.iter().map(|b| b.len()).sum();
    assert!(
        over > 0,
        "def2-TZVP should have nroots>5 classes; none found"
    );
    println!(
        "def2-TZVP/H2O: {over} of {total} quartets ({:.1}%) exceed the device nroots<=5 envelope",
        100.0 * over as f64 / total as f64
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 34-B — batched throughput.
//
// The per-quartet numbers above measure `eval_raw`, which pays a planner pass,
// twelve buffer allocations, a kernel dispatch and a blocking readback for
// every shell quartet. `evaluate_2e_quartet_batch` pays those once per launch
// class instead. Both engines still run the identical work-list, and the
// batched values are compared against the reference before any timing is
// printed — the same honesty rules as the benchmark above.
// ─────────────────────────────────────────────────────────────────────────────

/// Build the batch shell table from raw `atm`/`bas`/`env`.
///
/// The libcint env coefficient block is column-major (`env[ptr + c*nprim + p]`);
/// `BatchShell::coefficients` is primitive-major, matching `cintx_compat::raw`
/// (WR-03).
fn batch_shells(arrays: &RawArrays) -> Vec<cintx_cubecl::kernels::two_electron::BatchShell> {
    use cintx_compat::raw::{
        ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
    };
    (0..arrays.nbas())
        .map(|shell| {
            let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
            let nprim = record[NPRIM_OF] as usize;
            let nctr = record[NCTR_OF] as usize;
            let exp_ptr = record[PTR_EXP] as usize;
            let coeff_ptr = record[PTR_COEFF] as usize;
            let coord_ptr = arrays.atm[record[ATOM_OF] as usize * 6 + PTR_COORD] as usize;
            let mut coefficients = vec![0.0_f64; nprim * nctr];
            for c in 0..nctr {
                for p in 0..nprim {
                    coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
                }
            }
            cintx_cubecl::kernels::two_electron::BatchShell {
                l: record[ANG_OF] as u8,
                nprim: nprim as u32,
                nctr: nctr as u32,
                exponents: arrays.env[exp_ptr..exp_ptr + nprim].to_vec(),
                coefficients,
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect()
}

fn run_batch_case(label: &str, molecule: &Molecule, tolerance: f64) {
    use cintx_cubecl::backend::ResolvedBackend;
    use cintx_cubecl::kernels::two_electron::evaluate_2e_quartet_batch;
    use cintx_runtime::{BackendIntent, BackendKind};

    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let pairs = enumerate_pairs(&basis);
    let quartets = enumerate_quartets(&pairs);

    let mut vendor = VendorEngine { arrays: &arrays };
    let table = build_schwarz_table(&basis, &pairs, &mut vendor).expect("schwarz table");
    let (kept, _) = screen_quartets(&quartets, &table, tolerance);

    println!("\n{}", "=".repeat(100));
    println!("{label}  [BATCHED]");
    println!("{}", "=".repeat(100));
    println!(
        "  shells={}  quartets(8-fold)={}  kept={}  tol={:.0e}",
        basis.nbas(),
        quartets.len(),
        kept.len(),
        tolerance
    );

    // Reference: libcint over the identical list.
    let mut lengths = Vec::with_capacity(kept.len());
    let mut total = 0_usize;
    for &quartet in &kept {
        let len = cintx_driver::execute::block_len(&basis, quartet);
        lengths.push(len);
        total += len;
    }
    let repeats = bench_repeats();
    let mut reference = vec![0.0_f64; total];
    let mut ref_secs = f64::INFINITY;
    for _ in 0..repeats {
        let ref_start = std::time::Instant::now();
        {
            let mut cursor = 0;
            for (index, &quartet) in kept.iter().enumerate() {
                let len = lengths[index];
                vendor.eval(&quartet.shls(), &mut reference[cursor..cursor + len]);
                cursor += len;
            }
        }
        ref_secs = ref_secs.min(ref_start.elapsed().as_secs_f64());
    }

    let shells = batch_shells(&arrays);
    let list: Vec<[u32; 4]> = kept
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    // Warm-up: pay the per-class CubeCL specialization once, outside the timer.
    let warm_start = std::time::Instant::now();
    let _ = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("batched warm-up");
    let warm_secs = warm_start.elapsed().as_secs_f64();

    let mut act_secs = f64::INFINITY;
    let mut batched = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("batched 2e");
    for _ in 0..repeats {
        let act_start = std::time::Instant::now();
        batched = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("batched 2e");
        act_secs = act_secs.min(act_start.elapsed().as_secs_f64());
    }

    let mut max_diff = 0.0_f64;
    let mut mismatches = 0_usize;
    {
        let mut cursor = 0;
        for (index, &len) in lengths.iter().enumerate() {
            let start = batched.offsets[index];
            for element in 0..len {
                let diff = (reference[cursor + element] - batched.values[start + element]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-9 {
                    mismatches += 1;
                }
            }
            cursor += len;
        }
    }

    println!(
        "  launches={}  l-classes={}  readbacks={}  transfer={} KiB  \
         (vs {} launches per-quartet)",
        batched.stats.kernel_launch_count,
        batched.stats.launch_classes,
        batched.stats.readback_count,
        batched.stats.transfer_bytes / 1024,
        list.len()
    );
    println!(
        "  merge factor {:.2}x  widest G slab {} B/slot",
        batched.stats.launch_classes as f64 / batched.stats.kernel_launch_count.max(1) as f64,
        batched.stats.max_g_slab_bytes,
    );
    println!(
        "  first-call cost incl. per-class specialization: {warm_secs:.4} s  \
         (timed runs: best of {repeats})"
    );
    println!("  max|diff| vs vendor={max_diff:.3e}  mismatched elements={mismatches}");
    println!(
        "  last run split: backend dispatch {:.3} ms  host cart->sph {:.3} ms",
        batched.stats.dispatch_ns as f64 / 1e6,
        batched.stats.host_transform_ns as f64 / 1e6,
    );
    if let Some(split) = cintx_cubecl::transform::profile::format_split(&batched.stats) {
        println!("  {split}");
    }
    println!(
        "\n  {:<34} {:>12} {:>16} {:>16}",
        "engine", "wall (s)", "quartets/s", "us/quartet"
    );
    for (name, secs) in [
        ("libcint 6.1.3 (C, 1 thread)", ref_secs),
        ("cintx CubeCL batched (cpu)", act_secs),
    ] {
        println!(
            "  {:<34} {:>12.5} {:>16.1} {:>16.3}",
            name,
            secs,
            list.len() as f64 / secs.max(f64::MIN_POSITIVE),
            secs * 1e6 / list.len().max(1) as f64
        );
    }
    if mismatches == 0 {
        println!(
            "\n  VERDICT: batched cintx is {:.2}x {} than libcint on this workload.",
            if act_secs > ref_secs {
                act_secs / ref_secs
            } else {
                ref_secs / act_secs
            },
            if act_secs > ref_secs {
                "SLOWER"
            } else {
                "FASTER"
            }
        );
    } else {
        println!(
            "\n  VERDICT: NOT COMPARABLE — {mismatches} elements mismatched. \
             Speed is not reported for an incorrect run."
        );
    }

    // The batched case bucketes only for the record: `evaluate_2e_quartet_batch`
    // does its own class grouping internally, and re-deriving it here would be a
    // second opinion rather than a measurement. The bucket rows describe the
    // work list, and `launches`/`launch_classes` describe what the backend did
    // with it — D2.2's consolidation check reads the two together.
    let buckets = bucket_quartets(&basis, &kept);
    record(json!({
        "case": label,
        "mode": "batched",
        "tolerance": tolerance,
        "shells": basis.nbas(),
        "quartets_enumerated": quartets.len(),
        "quartets_kept": kept.len(),
        "buckets": buckets.len(),
        "envelope": envelope_split(&buckets),
        "bucket_rows": bucket_rows(&basis, &buckets),
        "kernel_launch_count": batched.stats.kernel_launch_count,
        "launch_classes": batched.stats.launch_classes,
        "readback_count": batched.stats.readback_count,
        "transfer_bytes": batched.stats.transfer_bytes,
        "max_g_slab_bytes": batched.stats.max_g_slab_bytes,
        "warmup_seconds": warm_secs,
        "libcint_seconds": ref_secs,
        "cintx_seconds": act_secs,
        "dispatch_ns": batched.stats.dispatch_ns,
        "host_transform_ns": batched.stats.host_transform_ns,
        "max_abs_diff_vs_vendor": max_diff,
        "mismatched_elements": mismatches,
        "comparable": mismatches == 0,
    }));
}

#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_batched_throughput() {
    run_batch_case(
        "H2O / def2-SVP  (unscreened)",
        &water(StandardBasis::Def2Svp),
        0.0,
    );
    run_batch_case(
        "H2O / def2-SVP  (screened 1e-10)",
        &water(StandardBasis::Def2Svp),
        1e-10,
    );
    run_batch_case(
        "CH4 / def2-SVP  (screened 1e-10)",
        &methane(StandardBasis::Def2Svp),
        1e-10,
    );
    run_batch_case(
        "SO2 / def2-SVP  (screened 1e-10)",
        &sulfur_dioxide(StandardBasis::Def2Svp),
        1e-10,
    );
    if std::env::var("CINTX_BENCH_SCOPE").as_deref() == Ok("full") {
        // TZVP through the batch path needs the extended device Rys ceiling for
        // its nroots 6-7 classes (D1). Without the feature those classes are
        // refused, and the case would report NOT COMPARABLE rather than a
        // number — correct, but not worth the minutes it costs by default.
        run_batch_case(
            "H2O / def2-TZVP (screened 1e-10)",
            &water(StandardBasis::Def2Tzvp),
            1e-10,
        );
        run_batch_case(
            "SO2 / def2-TZVP (screened 1e-10)",
            &sulfur_dioxide(StandardBasis::Def2Tzvp),
            1e-10,
        );
    } else {
        println!("\n(def2-TZVP batch cases skipped; set CINTX_BENCH_SCOPE=full.)");
    }

    flush_artifact("def2_batched_throughput");
}
