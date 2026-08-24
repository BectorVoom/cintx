//! Phase 35 — the batched shell-*pair* paths (`int1e_*` and `int2c2e`) must
//! match vendored libcint exactly, and must actually amortize the launch.
//!
//! `evaluate_1e_pair_batch` / `evaluate_2c2e_pair_batch` group a whole
//! shell-pair work list into `(li, lj)` launch classes and run one dispatch per
//! class. That is a different launch
//! topology from the per-pair compatibility route (grid over pairs, per-slot
//! G-tensor slab, flattened basis read through an index table) reaching the same
//! arithmetic, so the bar is identity against the two references that matter:
//!
//! 1. **vendored libcint 6.1.3** — the compatibility contract;
//! 2. **cintx's own per-pair `eval_raw`** — so a batching regression cannot hide
//!    behind a tolerance the single path also fails.
//!
//! The launch-count assertion is what makes the throughput claim auditable: a
//! batch that quietly fell back to one dispatch per pair would still be correct
//! and would still be slow.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_pair_batch_parity`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{
    ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
    RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::kernels::center_2c2e::evaluate_2c2e_pair_batch;
use cintx_cubecl::kernels::center_3c2e::evaluate_3c2e_triple_batch;
use cintx_cubecl::kernels::one_electron::{BatchAtom, OneEOperator, evaluate_1e_pair_batch};
use cintx_cubecl::kernels::two_electron::BatchShell;
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// Build the batch shell table from the raw `atm`/`bas`/`env` arrays.
///
/// The libcint env coefficient block is contraction-major
/// (`env[ptr + c*nprim + p]`); `BatchShell::coefficients` is primitive-major,
/// the layout the kernel reads. The transpose here mirrors `cintx_compat::raw`.
fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    let nbas = arrays.nbas();
    let mut shells = Vec::with_capacity(nbas);
    for shell in 0..nbas {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;
        let atom = record[ATOM_OF] as usize;
        let coord_ptr = arrays.atm[atom * 6 + PTR_COORD] as usize;

        let mut coefficients = vec![0.0_f64; nprim * nctr];
        for c in 0..nctr {
            for p in 0..nprim {
                coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
            }
        }

        shells.push(BatchShell {
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
        });
    }
    shells
}

fn batch_atoms(arrays: &RawArrays) -> Vec<BatchAtom> {
    (0..arrays.natm())
        .map(|atom| {
            let coord_ptr = arrays.atm[atom * 6 + PTR_COORD] as usize;
            BatchAtom {
                charge: f64::from(arrays.atm[atom * 6 + CHARGE_OF]),
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect()
}

fn water(basis: StandardBasis) -> Molecule {
    Molecule::new(
        vec![
            AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
        ],
        basis,
    )
}

type VendorFn = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;

fn case(operator: OneEOperator) -> (RawApiId, VendorFn) {
    match operator {
        OneEOperator::Overlap => (
            RawApiId::INT1E_OVLP_SPH,
            vendor_ffi::vendor_int1e_ovlp_sph as VendorFn,
        ),
        OneEOperator::Kinetic => (
            RawApiId::INT1E_KIN_SPH,
            vendor_ffi::vendor_int1e_kin_sph as VendorFn,
        ),
        OneEOperator::Nuclear => (
            RawApiId::INT1E_NUC_SPH,
            vendor_ffi::vendor_int1e_nuc_sph as VendorFn,
        ),
    }
}

fn nsph(l: u8) -> usize {
    2 * l as usize + 1
}

/// Every def2-SVP shell pair, all three scalar 1e operators, batched — against
/// libcint and against cintx's own per-pair route.
#[test]
fn def2_svp_1e_batch_matches_vendor_and_per_pair() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let atoms = batch_atoms(&arrays);
    let nbas = shells.len();
    assert!(nbas > 1);

    let list: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    for operator in [
        OneEOperator::Overlap,
        OneEOperator::Kinetic,
        OneEOperator::Nuclear,
    ] {
        let (api, vendor) = case(operator);
        let batched = evaluate_1e_pair_batch(&backend, operator, &shells, &atoms, &list)
            .unwrap_or_else(|error| panic!("{} batch failed: {error}", operator.symbol()));

        // One dispatch per Rys order, not per `(li,lj)` class and not per pair.
        //
        // Phase 35 batched by l-class; Task 35-M2 merged those, because `nroots`
        // (with the caller-fixed `op_kind`) is the kernel's whole comptime
        // specialization and `li`/`lj` were already runtime scalars. Overlap and
        // kinetic are not Rys quadratures at all, so they collapse to a single
        // dispatch; nuclear keeps one per distinct `(li+lj)/2 + 1`.
        let classes: std::collections::BTreeSet<[u8; 2]> = list
            .iter()
            .map(|p| [shells[p[0] as usize].l, shells[p[1] as usize].l])
            .collect();
        let dispatches: std::collections::BTreeSet<usize> = match operator {
            OneEOperator::Nuclear => classes
                .iter()
                .map(|&[li, lj]| (li as usize + lj as usize) / 2 + 1)
                .collect(),
            _ => std::iter::once(1).collect(),
        };
        assert_eq!(
            batched.stats.launch_classes,
            classes.len(),
            "{}: every (li,lj) class must be accounted for",
            operator.symbol()
        );
        assert_eq!(
            batched.stats.kernel_launch_count,
            dispatches.len(),
            "{}: expected one dispatch per Rys order",
            operator.symbol()
        );
        assert!(
            batched.stats.kernel_launch_count <= classes.len(),
            "{}: merging must not increase the dispatch count",
            operator.symbol()
        );
        assert_eq!(
            batched.stats.readback_count,
            batched.stats.kernel_launch_count
        );
        assert!(
            batched.stats.kernel_launch_count < list.len(),
            "{}: batching must reduce the launch count below the pair count",
            operator.symbol()
        );

        let mut vendor_mismatches = 0_usize;
        let mut single_mismatches = 0_usize;
        let mut max_vendor_diff = 0.0_f64;
        let mut first_report = String::new();

        for (index, pair) in list.iter().enumerate() {
            let shls = [pair[0] as i32, pair[1] as i32];
            let len = nsph(shells[pair[0] as usize].l)
                * shells[pair[0] as usize].nctr as usize
                * nsph(shells[pair[1] as usize].l)
                * shells[pair[1] as usize].nctr as usize;

            let mut expected = vec![0.0_f64; len];
            vendor(
                &mut expected,
                &shls,
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );

            let mut single = vec![0.0_f64; len];
            unsafe {
                eval_raw(
                    api,
                    Some(&mut single),
                    None,
                    &shls,
                    &arrays.atm,
                    &arrays.bas,
                    &arrays.env,
                    None,
                    None,
                )
            }
            .expect("per-pair evaluation");

            let start = batched.offsets[index];
            let actual = &batched.values[start..start + len];
            for element in 0..len {
                let diff = (expected[element] - actual[element]).abs();
                if diff > max_vendor_diff {
                    max_vendor_diff = diff;
                }
                if diff > 1e-12 {
                    vendor_mismatches += 1;
                    if first_report.is_empty() {
                        first_report = format!(
                            "{} shls={shls:?} element={element} vendor={:.17e} batch={:.17e}",
                            operator.symbol(),
                            expected[element],
                            actual[element]
                        );
                    }
                }
                if single[element].to_bits() != actual[element].to_bits() {
                    single_mismatches += 1;
                    if first_report.is_empty() {
                        first_report = format!(
                            "{} shls={shls:?} element={element} per-pair={:.17e} batch={:.17e}",
                            operator.symbol(),
                            single[element],
                            actual[element]
                        );
                    }
                }
            }
        }

        assert_eq!(
            single_mismatches,
            0,
            "{}: batched path must be BIT-identical to the per-pair path; first: {first_report}",
            operator.symbol()
        );
        assert_eq!(
            vendor_mismatches,
            0,
            "{}: batched path must match vendored libcint (max |diff| {max_vendor_diff:.3e}); \
             first: {first_report}",
            operator.symbol()
        );
    }
}

/// An empty work list is a valid, empty batch — not a panic and not a dispatch.
#[test]
fn empty_1e_batch_is_a_no_op() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let output = evaluate_1e_pair_batch(&backend, OneEOperator::Overlap, &shells, &[], &[])
        .expect("empty batch");
    assert!(output.values.is_empty());
    assert_eq!(output.stats.kernel_launch_count, 0);
}

/// Phase 35 acceptance — batching must be at least 10x the per-pair CubeCL
/// throughput on 1e.
///
/// Both sides run the identical pair list on the identical basis, and the
/// batched values are compared against the per-pair values before any timing is
/// reported: a speed number for a wrong answer is not a speed number. libcint is
/// timed on the same list purely as a scale reference.
#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_1e_batched_throughput() {
    let repeats: usize = std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(5);

    for basis_set in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        let molecule = water(basis_set);
        let arrays = to_raw_arrays(&molecule).expect("raw arrays");
        let shells = batch_shells(&arrays);
        let atoms = batch_atoms(&arrays);
        let nbas = shells.len();
        let list: Vec<[u32; 2]> = (0..nbas)
            .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
            .collect();

        let backend = ResolvedBackend::from_intent(&BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend");

        println!("\n{}", "=".repeat(96));
        println!(
            "H2O / {basis_set:?}  int1e_*  ({nbas} shells, {} pairs)",
            list.len()
        );
        println!("{}", "=".repeat(96));

        for operator in [
            OneEOperator::Overlap,
            OneEOperator::Kinetic,
            OneEOperator::Nuclear,
        ] {
            let (api, vendor) = case(operator);

            let lengths: Vec<usize> = list
                .iter()
                .map(|p| {
                    nsph(shells[p[0] as usize].l)
                        * shells[p[0] as usize].nctr as usize
                        * nsph(shells[p[1] as usize].l)
                        * shells[p[1] as usize].nctr as usize
                })
                .collect();
            let total: usize = lengths.iter().sum();

            // Reference: libcint over the identical list.
            let mut reference = vec![0.0_f64; total];
            let mut vendor_secs = f64::INFINITY;
            for _ in 0..repeats {
                let start = std::time::Instant::now();
                let mut cursor = 0;
                for (index, pair) in list.iter().enumerate() {
                    let len = lengths[index];
                    vendor(
                        &mut reference[cursor..cursor + len],
                        &[pair[0] as i32, pair[1] as i32],
                        &arrays.atm,
                        arrays.natm() as i32,
                        &arrays.bas,
                        arrays.nbas() as i32,
                        &arrays.env,
                    );
                    cursor += len;
                }
                vendor_secs = vendor_secs.min(start.elapsed().as_secs_f64());
            }

            // cintx, per pair — the shape Phase 35 replaces.
            let mut per_pair = vec![0.0_f64; total];
            let mut per_pair_secs = f64::INFINITY;
            for _ in 0..repeats {
                let start = std::time::Instant::now();
                let mut cursor = 0;
                for (index, pair) in list.iter().enumerate() {
                    let len = lengths[index];
                    unsafe {
                        eval_raw(
                            api,
                            Some(&mut per_pair[cursor..cursor + len]),
                            None,
                            &[pair[0] as i32, pair[1] as i32],
                            &arrays.atm,
                            &arrays.bas,
                            &arrays.env,
                            None,
                            None,
                        )
                    }
                    .expect("per-pair evaluation");
                    cursor += len;
                }
                per_pair_secs = per_pair_secs.min(start.elapsed().as_secs_f64());
            }

            // cintx, batched. Warm-up pays the per-class specialization once.
            let mut batched =
                evaluate_1e_pair_batch(&backend, operator, &shells, &atoms, &list).expect("batch");
            let mut batch_secs = f64::INFINITY;
            for _ in 0..repeats {
                let start = std::time::Instant::now();
                batched = evaluate_1e_pair_batch(&backend, operator, &shells, &atoms, &list)
                    .expect("batch");
                batch_secs = batch_secs.min(start.elapsed().as_secs_f64());
            }

            let mut max_diff = 0.0_f64;
            let mut cursor = 0;
            for (index, &len) in lengths.iter().enumerate() {
                let start = batched.offsets[index];
                for element in 0..len {
                    let diff =
                        (reference[cursor + element] - batched.values[start + element]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                cursor += len;
            }
            assert!(
                max_diff < 1e-12,
                "{}: batched result disagrees with libcint by {max_diff:.3e}; \
                 speed is not reported for an incorrect run",
                operator.symbol()
            );

            let speedup = per_pair_secs / batch_secs;
            println!(
                "  {:<16} libcint {:>9.5} s   cintx per-pair {:>9.5} s   \
                 cintx batched {:>9.5} s ({} launches)   speed-up {:>6.1}x",
                operator.symbol(),
                vendor_secs,
                per_pair_secs,
                batch_secs,
                batched.stats.kernel_launch_count,
                speedup,
            );
        }
    }
}

/// Phase 35 — the batched `int2c2e` path, against libcint and against cintx's
/// own per-pair route.
///
/// def2-SVP's ordinary AO shells stand in for an auxiliary basis here: `int2c2e`
/// does not care which basis a shell came from, and using the shells that are
/// already in the fixture keeps the test independent of whether def2/J has been
/// added to `cintx-basis` yet.
#[test]
fn def2_svp_2c2e_batch_matches_vendor_and_per_pair() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = shells.len();

    // The device 2c2e kernel caps at nroots 5, i.e. li + lk <= 8; def2-SVP water
    // tops out at d shells, so every pair is inside the envelope.
    let list: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |k| [i as u32, k as u32]))
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let batched = evaluate_2c2e_pair_batch(&backend, &shells, &list).expect("2c2e batch");

    // One dispatch per Rys order (Task 35-M2), not per `(li,lk)` class.
    let classes: std::collections::BTreeSet<[u8; 2]> = list
        .iter()
        .map(|p| [shells[p[0] as usize].l, shells[p[1] as usize].l])
        .collect();
    let dispatches: std::collections::BTreeSet<usize> = classes
        .iter()
        .map(|&[li, lk]| (li as usize + lk as usize) / 2 + 1)
        .collect();
    assert_eq!(batched.stats.launch_classes, classes.len());
    assert_eq!(batched.stats.kernel_launch_count, dispatches.len());
    assert!(
        batched.stats.kernel_launch_count < classes.len(),
        "merging must dispatch fewer times than there are classes: {} vs {}",
        batched.stats.kernel_launch_count,
        classes.len()
    );
    assert!(batched.stats.kernel_launch_count < list.len());

    let mut vendor_mismatches = 0_usize;
    let mut single_mismatches = 0_usize;
    let mut max_vendor_diff = 0.0_f64;
    let mut first_report = String::new();

    for (index, pair) in list.iter().enumerate() {
        let shls = [pair[0] as i32, pair[1] as i32];
        let len = nsph(shells[pair[0] as usize].l)
            * shells[pair[0] as usize].nctr as usize
            * nsph(shells[pair[1] as usize].l)
            * shells[pair[1] as usize].nctr as usize;

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2c2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );

        let mut single = vec![0.0_f64; len];
        unsafe {
            eval_raw(
                RawApiId::INT2C2E_SPH,
                Some(&mut single),
                None,
                &shls,
                &arrays.atm,
                &arrays.bas,
                &arrays.env,
                None,
                None,
            )
        }
        .expect("per-pair 2c2e");

        let start = batched.offsets[index];
        let actual = &batched.values[start..start + len];
        for element in 0..len {
            let diff = (expected[element] - actual[element]).abs();
            if diff > max_vendor_diff {
                max_vendor_diff = diff;
            }
            if diff > 1e-12 {
                vendor_mismatches += 1;
                if first_report.is_empty() {
                    first_report = format!(
                        "shls={shls:?} element={element} vendor={:.17e} batch={:.17e}",
                        expected[element], actual[element]
                    );
                }
            }
            if single[element].to_bits() != actual[element].to_bits() {
                single_mismatches += 1;
                if first_report.is_empty() {
                    first_report = format!(
                        "shls={shls:?} element={element} per-pair={:.17e} batch={:.17e}",
                        single[element], actual[element]
                    );
                }
            }
        }
    }

    assert_eq!(
        single_mismatches, 0,
        "batched 2c2e must be BIT-identical to the per-pair path; first: {first_report}"
    );
    assert_eq!(
        vendor_mismatches, 0,
        "batched 2c2e must match vendored libcint (max |diff| {max_vendor_diff:.3e}); \
         first: {first_report}"
    );
}

/// Phase 35's highest-priority family — the batched `int3c2e` path, against
/// libcint and against cintx's own per-triple route.
///
/// The `swap_ij` canonicalization is what makes 3c2e different from the other
/// batched families: the kernel only evaluates `li >= lj`, so half the classes
/// are transposed on the way out. Enumerating every `(i, j, k)` triple — not
/// just the canonical half — is what puts that path under test.
#[test]
fn def2_svp_3c2e_batch_matches_vendor_and_per_triple() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = shells.len();

    // The device 3c2e kernel caps at nroots 5, i.e. li + lj + lk <= 8. def2-SVP
    // water tops out at d shells, so a (d,d,d) triple would need nroots 4 — well
    // inside the envelope, and every triple is admissible.
    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("3c2e batch");

    let classes: std::collections::BTreeSet<[u8; 3]> = list
        .iter()
        .map(|t| {
            [
                shells[t[0] as usize].l,
                shells[t[1] as usize].l,
                shells[t[2] as usize].l,
            ]
        })
        .collect();
    // One dispatch per Rys order (Task 35-M2), not per `(li,lj,lk)` class.
    // `nroots` uses the canonical `li >= lj` ordering, but `(li + lj)` — and so
    // the Rys order — is invariant under that swap.
    let dispatches: std::collections::BTreeSet<usize> = classes
        .iter()
        .map(|&[li, lj, lk]| (li as usize + lj as usize + lk as usize) / 2 + 1)
        .collect();
    assert_eq!(batched.stats.launch_classes, classes.len());
    assert_eq!(batched.stats.kernel_launch_count, dispatches.len());
    assert!(
        batched.stats.kernel_launch_count < classes.len(),
        "merging must dispatch fewer times than there are classes: {} vs {}",
        batched.stats.kernel_launch_count,
        classes.len()
    );
    assert!(batched.stats.kernel_launch_count < list.len());
    // Both halves of the swap must be represented, or the transpose path is
    // untested.
    assert!(
        classes.iter().any(|c| c[0] < c[1]) && classes.iter().any(|c| c[0] > c[1]),
        "the fixture must contain both li<lj and li>lj classes"
    );

    let mut vendor_mismatches = 0_usize;
    let mut single_mismatches = 0_usize;
    let mut max_vendor_diff = 0.0_f64;
    let mut first_report = String::new();

    for (index, triple) in list.iter().enumerate() {
        let shls = [triple[0] as i32, triple[1] as i32, triple[2] as i32];
        let len: usize = triple
            .iter()
            .map(|&s| nsph(shells[s as usize].l) * shells[s as usize].nctr as usize)
            .product();

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int3c2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );

        let mut single = vec![0.0_f64; len];
        unsafe {
            eval_raw(
                RawApiId::INT3C2E_SPH,
                Some(&mut single),
                None,
                &shls,
                &arrays.atm,
                &arrays.bas,
                &arrays.env,
                None,
                None,
            )
        }
        .expect("per-triple 3c2e");

        let start = batched.offsets[index];
        let actual = &batched.values[start..start + len];
        for element in 0..len {
            let diff = (expected[element] - actual[element]).abs();
            if diff > max_vendor_diff {
                max_vendor_diff = diff;
            }
            if diff > 1e-12 {
                vendor_mismatches += 1;
                if first_report.is_empty() {
                    first_report = format!(
                        "shls={shls:?} element={element} vendor={:.17e} batch={:.17e}",
                        expected[element], actual[element]
                    );
                }
            }
            if single[element].to_bits() != actual[element].to_bits() {
                single_mismatches += 1;
                if first_report.is_empty() {
                    first_report = format!(
                        "shls={shls:?} element={element} per-triple={:.17e} batch={:.17e}",
                        single[element], actual[element]
                    );
                }
            }
        }
    }

    assert_eq!(
        single_mismatches, 0,
        "batched 3c2e must be BIT-identical to the per-triple path; first: {first_report}"
    );
    assert_eq!(
        vendor_mismatches, 0,
        "batched 3c2e must match vendored libcint (max |diff| {max_vendor_diff:.3e}); \
         first: {first_report}"
    );
}

/// The combination the def2 fixtures cannot reach: a **general contraction**
/// (`nctr > 1`) evaluated through the batched 3c2e path, over triples that
/// exercise both sides of the `swap_ij` canonicalization.
///
/// def2-SVP and def2-TZVP are fully segmented, so every def2 test above runs
/// with `nctr == 1` — where the contraction-block index is trivially 0 and a
/// swapped block stride is indistinguishable from an unswapped one. This fixture
/// is a p shell with `nprim = 3, nctr = 2` beside an s and a d shell, so
/// `(s, p, ...)` classes swap and `(p, s, ...)` classes do not, and both carry
/// four contraction blocks.
#[test]
fn general_contraction_3c2e_batch_matches_vendor() {
    use cintx_compat::raw::{ATM_SLOTS, NUC_MOD_OF, POINT_NUC, PTR_ENV_START, PTR_ZETA};

    let mut env = vec![0.0_f64; PTR_ENV_START];
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let a_coord = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, 0.0]);
    let b_coord = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.3, 1.4]);

    // Deliberately asymmetric coefficients, contraction-major in `env`
    // (`coeff[ic * nprim + ip]`), so a transposed or collapsed read differs.
    let p_exps = [6.0_f64, 1.5, 0.4];
    let p_coeffs = [0.20_f64, 0.55, 0.30, -0.10, 0.35, 0.80];
    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exps);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeffs);

    let s_exp = [0.9_f64];
    let s_coeff = [1.0_f64];
    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let d_exp = [1.3_f64];
    let d_coeff = [1.0_f64];
    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (index, ptr) in [(0_usize, a_coord), (1, b_coord)] {
        atm[index * ATM_SLOTS + CHARGE_OF] = 1;
        atm[index * ATM_SLOTS + PTR_COORD] = ptr;
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    // shell 0: p, nprim=3, nctr=2 on atom 0
    // shell 1: s, nprim=1, nctr=1 on atom 1
    // shell 2: d, nprim=1, nctr=1 on atom 0
    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    for (shell, atom, l, nprim, nctr, e, c) in [
        (0_usize, 0_i32, 1_i32, 3_i32, 2_i32, p_exp_ptr, p_coeff_ptr),
        (1, 1, 0, 1, 1, s_exp_ptr, s_coeff_ptr),
        (2, 0, 2, 1, 1, d_exp_ptr, d_coeff_ptr),
    ] {
        bas[shell * BAS_SLOTS + ATOM_OF] = atom;
        bas[shell * BAS_SLOTS + ANG_OF] = l;
        bas[shell * BAS_SLOTS + NPRIM_OF] = nprim;
        bas[shell * BAS_SLOTS + NCTR_OF] = nctr;
        bas[shell * BAS_SLOTS + PTR_EXP] = e;
        bas[shell * BAS_SLOTS + PTR_COEFF] = c;
    }

    let arrays = RawArrays { atm, bas, env };
    let shells = batch_shells(&arrays);
    let nbas = shells.len();
    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");
    let batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("3c2e batch");

    let mut max_diff = 0.0_f64;
    let mut first_report = String::new();
    for (index, triple) in list.iter().enumerate() {
        let shls = [triple[0] as i32, triple[1] as i32, triple[2] as i32];
        let len: usize = triple
            .iter()
            .map(|&s| nsph(shells[s as usize].l) * shells[s as usize].nctr as usize)
            .product();
        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int3c2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );
        let start = batched.offsets[index];
        for element in 0..len {
            let diff = (expected[element] - batched.values[start + element]).abs();
            if diff > max_diff {
                max_diff = diff;
                first_report = format!(
                    "shls={shls:?} element={element} vendor={:.17e} batch={:.17e}",
                    expected[element],
                    batched.values[start + element]
                );
            }
        }
    }

    assert!(
        max_diff < 1e-12,
        "general-contraction 3c2e batch disagrees with vendored libcint by \
         {max_diff:.3e}; worst: {first_report}"
    );
}

/// Phase 35 — batched vs per-tuple throughput for `int2c2e` and `int3c2e`.
///
/// Same rules as the 1e benchmark above: both engines run the identical work
/// list, the batched values are checked against libcint before any timing is
/// reported, and libcint is timed on the same list as a scale reference.
#[test]
#[ignore = "throughput benchmark; run explicitly in release with --ignored"]
fn def2_2c2e_3c2e_batched_throughput() {
    let repeats: usize = std::env::var("CINTX_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(5);

    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = shells.len();
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    println!("\n{}", "=".repeat(96));
    println!("H2O / def2-SVP  int2c2e / int3c2e  ({nbas} shells)");
    println!("{}", "=".repeat(96));

    // ── int2c2e ─────────────────────────────────────────────────────────────
    {
        let list: Vec<[u32; 2]> = (0..nbas)
            .flat_map(|i| (0..nbas).map(move |k| [i as u32, k as u32]))
            .collect();
        let lengths: Vec<usize> = list
            .iter()
            .map(|p| {
                p.iter()
                    .map(|&s| nsph(shells[s as usize].l) * shells[s as usize].nctr as usize)
                    .product()
            })
            .collect();
        let total: usize = lengths.iter().sum();

        let mut reference = vec![0.0_f64; total];
        let mut vendor_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let mut cursor = 0;
            for (index, p) in list.iter().enumerate() {
                vendor_ffi::vendor_int2c2e_sph(
                    &mut reference[cursor..cursor + lengths[index]],
                    &[p[0] as i32, p[1] as i32],
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                );
                cursor += lengths[index];
            }
            vendor_secs = vendor_secs.min(start.elapsed().as_secs_f64());
        }

        let mut scratch = vec![0.0_f64; total];
        let mut per_tuple_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let mut cursor = 0;
            for (index, p) in list.iter().enumerate() {
                unsafe {
                    eval_raw(
                        RawApiId::INT2C2E_SPH,
                        Some(&mut scratch[cursor..cursor + lengths[index]]),
                        None,
                        &[p[0] as i32, p[1] as i32],
                        &arrays.atm,
                        &arrays.bas,
                        &arrays.env,
                        None,
                        None,
                    )
                }
                .expect("per-pair 2c2e");
                cursor += lengths[index];
            }
            per_tuple_secs = per_tuple_secs.min(start.elapsed().as_secs_f64());
        }

        let mut batched = evaluate_2c2e_pair_batch(&backend, &shells, &list).expect("batch");
        let mut batch_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            batched = evaluate_2c2e_pair_batch(&backend, &shells, &list).expect("batch");
            batch_secs = batch_secs.min(start.elapsed().as_secs_f64());
        }

        let mut max_diff = 0.0_f64;
        let mut cursor = 0;
        for (index, &len) in lengths.iter().enumerate() {
            let start = batched.offsets[index];
            for element in 0..len {
                max_diff = max_diff
                    .max((reference[cursor + element] - batched.values[start + element]).abs());
            }
            cursor += len;
        }
        assert!(
            max_diff < 1e-12,
            "int2c2e batch disagrees by {max_diff:.3e}"
        );

        println!(
            "  {:<16} libcint {:>9.5} s   cintx per-tuple {:>9.5} s   \
             cintx batched {:>9.5} s ({} launches, {} tuples)   speed-up {:>6.1}x",
            "int2c2e_sph",
            vendor_secs,
            per_tuple_secs,
            batch_secs,
            batched.stats.kernel_launch_count,
            list.len(),
            per_tuple_secs / batch_secs,
        );
    }

    // ── int3c2e ─────────────────────────────────────────────────────────────
    {
        let list: Vec<[u32; 3]> = (0..nbas)
            .flat_map(|i| {
                (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
            })
            .collect();
        let lengths: Vec<usize> = list
            .iter()
            .map(|t| {
                t.iter()
                    .map(|&s| nsph(shells[s as usize].l) * shells[s as usize].nctr as usize)
                    .product()
            })
            .collect();
        let total: usize = lengths.iter().sum();

        let mut reference = vec![0.0_f64; total];
        let mut vendor_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let mut cursor = 0;
            for (index, t) in list.iter().enumerate() {
                vendor_ffi::vendor_int3c2e_sph(
                    &mut reference[cursor..cursor + lengths[index]],
                    &[t[0] as i32, t[1] as i32, t[2] as i32],
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                );
                cursor += lengths[index];
            }
            vendor_secs = vendor_secs.min(start.elapsed().as_secs_f64());
        }

        let mut scratch = vec![0.0_f64; total];
        let mut per_tuple_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let mut cursor = 0;
            for (index, t) in list.iter().enumerate() {
                unsafe {
                    eval_raw(
                        RawApiId::INT3C2E_SPH,
                        Some(&mut scratch[cursor..cursor + lengths[index]]),
                        None,
                        &[t[0] as i32, t[1] as i32, t[2] as i32],
                        &arrays.atm,
                        &arrays.bas,
                        &arrays.env,
                        None,
                        None,
                    )
                }
                .expect("per-triple 3c2e");
                cursor += lengths[index];
            }
            per_tuple_secs = per_tuple_secs.min(start.elapsed().as_secs_f64());
        }

        let mut batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("batch");
        let mut batch_secs = f64::INFINITY;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            batched = evaluate_3c2e_triple_batch(&backend, &shells, &list).expect("batch");
            batch_secs = batch_secs.min(start.elapsed().as_secs_f64());
        }

        let mut max_diff = 0.0_f64;
        let mut cursor = 0;
        for (index, &len) in lengths.iter().enumerate() {
            let start = batched.offsets[index];
            for element in 0..len {
                max_diff = max_diff
                    .max((reference[cursor + element] - batched.values[start + element]).abs());
            }
            cursor += len;
        }
        assert!(
            max_diff < 1e-12,
            "int3c2e batch disagrees by {max_diff:.3e}"
        );

        println!(
            "  {:<16} libcint {:>9.5} s   cintx per-tuple {:>9.5} s   \
             cintx batched {:>9.5} s ({} launches, {} tuples)   speed-up {:>6.1}x",
            "int3c2e_sph",
            vendor_secs,
            per_tuple_secs,
            batch_secs,
            batched.stats.kernel_launch_count,
            list.len(),
            per_tuple_secs / batch_secs,
        );
    }
}
