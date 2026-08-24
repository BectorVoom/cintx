//! Task 34-B — the batched shell-quartet path must be *identical* to the
//! per-quartet path, not merely close to it.
//!
//! `evaluate_2e_quartet_batch` groups a whole work-list into launch classes and
//! runs one dispatch per class. That is a different kernel launch topology
//! (grid over quartets, per-cube G-tensor slab, flattened basis read through an
//! index table) reaching the same arithmetic, so the acceptance bar is byte
//! identity against the two references that matter:
//!
//! 1. **vendored libcint 6.1.3** — the compatibility contract;
//! 2. **cintx's own per-quartet `eval_raw`** — so a batching regression cannot
//!    hide behind a tolerance that the single path also fails.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_2e_batch_parity`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{
    ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP, RawApiId,
    eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::kernels::two_electron::{
    BatchShell, ResidentTwoEBasis, TwoEBatchOptions, evaluate_2e_quartet_batch,
    evaluate_2e_quartet_batch_resident, evaluate_2e_quartet_batch_with,
};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// Build the batch shell table from the raw `atm`/`bas`/`env` arrays.
///
/// The libcint env coefficient block is column-major (`env[ptr + c*nprim + p]`);
/// `BatchShell::coefficients` is primitive-major, the layout the kernel has
/// always read. The transpose here mirrors `cintx_compat::raw` (WR-03).
fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    let nbas = arrays.nbas();
    let mut shells = Vec::with_capacity(nbas);
    for shell in 0..nbas {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let l = record[ANG_OF] as u8;
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
            l,
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

/// The kernel's comptime signature for an l-quartet: `(ibase, kbase, nroots)`.
///
/// Mirrors `build_2e_shape` rather than calling it — `cintx-cubecl` does not
/// export the shape builder, and an independent transcription is what makes
/// this a check on the merge rather than a restatement of it. libcint picks the
/// HRR base by which side of a pair carries more angular momentum
/// (`g2e.c` `CINTinit_int2e_EnvVars`), and `nroots = (sum l)/2 + 1`.
fn launch_signature(li: u8, lj: u8, lk: u8, ll: u8) -> (usize, usize, usize) {
    // Strict `>`, as libcint has it — `li == lj` takes the `false` branch.
    let ibase = usize::from(li > lj);
    let kbase = usize::from(lk > ll);
    let nroots = (li as usize + lj as usize + lk as usize + ll as usize) / 2 + 1;
    (ibase, kbase, nroots)
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

/// Every def2-SVP quartet, evaluated in one batched call, must equal both
/// vendored libcint and cintx's own per-quartet path **bit for bit**.
#[test]
fn def2_svp_batch_matches_vendor_and_per_quartet() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    assert!(!quartets.is_empty(), "def2-SVP water must produce quartets");

    let shells = batch_shells(&arrays);
    let list: Vec<[u32; 4]> = quartets
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let batched = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("batched 2e");

    // One dispatch per *launch signature*, not per l-class and not per quartet.
    //
    // Task 34-B took the dispatch count from one-per-quartet to one-per-l-class;
    // Task 35-M1 takes it further, to one per `(ibase, kbase, nroots)`, because
    // those are the kernel's only comptime parameters and every other shape
    // scalar is read per quartet. The assertion below is the load-bearing half
    // of that claim — the other half is the bit-identity comparison after it,
    // which merging must not disturb.
    let classes: std::collections::BTreeSet<[u8; 4]> = list
        .iter()
        .map(|q| {
            [
                shells[q[0] as usize].l,
                shells[q[1] as usize].l,
                shells[q[2] as usize].l,
                shells[q[3] as usize].l,
            ]
        })
        .collect();
    let signatures: std::collections::BTreeSet<(usize, usize, usize)> = classes
        .iter()
        .map(|&[li, lj, lk, ll]| launch_signature(li, lj, lk, ll))
        .collect();
    assert_eq!(
        batched.stats.launch_classes,
        classes.len(),
        "every l-class in the list must be accounted for"
    );
    assert_eq!(
        batched.stats.kernel_launch_count,
        signatures.len(),
        "batched run must launch once per (ibase, kbase, nroots) signature"
    );
    assert!(
        batched.stats.kernel_launch_count < classes.len(),
        "merging must reduce dispatches below the l-class count: {} launches for {} classes",
        batched.stats.kernel_launch_count,
        classes.len()
    );
    assert_eq!(batched.stats.readback_count, signatures.len());
    assert_eq!(batched.stats.quartets, list.len());

    // Task 34-C (in-call half): the flattened basis is uploaded ONCE for the
    // whole run, not once per launch class. Bound the reported transfer by
    // "one basis + every class's quartet table" and check it is far below what
    // per-class re-upload would have cost.
    let basis_bytes: usize = shells
        .iter()
        .map(|shell| {
            (shell.nprim as usize + (shell.nprim * shell.nctr) as usize + 3)
                * std::mem::size_of::<f64>()
                + 4 * std::mem::size_of::<u32>()
        })
        .sum();
    // Six `u32` per quartet row, plus one 13-`u32` shape row and one `f64`
    // common factor per merged l-class.
    let table_bytes = list.len() * 6 * std::mem::size_of::<u32>()
        + classes.len() * (13 * std::mem::size_of::<u32>() + std::mem::size_of::<f64>());
    assert_eq!(
        batched.stats.transfer_bytes,
        basis_bytes + table_bytes,
        "transfer must be one basis upload plus the quartet and class tables"
    );
    assert!(
        batched.stats.transfer_bytes
            < basis_bytes * batched.stats.kernel_launch_count + table_bytes,
        "basis must not be re-uploaded per launch class"
    );
    assert!(
        batched.stats.kernel_launch_count < list.len(),
        "batching must reduce launches: {} launches for {} quartets",
        batched.stats.kernel_launch_count,
        list.len()
    );

    let mut vendor_mismatches = 0_usize;
    let mut single_mismatches = 0_usize;
    let mut max_vendor_diff = 0.0_f64;
    let mut first_report = String::new();

    for (index, quartet) in quartets.iter().enumerate() {
        let shls = quartet.shls();
        let len = cintx_driver::execute::block_len(&basis, *quartet);
        let start = batched.offsets[index];
        let actual = &batched.values[start..start + len];

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );

        let mut single = vec![0.0_f64; len];
        // SAFETY: `single` is sized from the same `bas` array `eval_raw` reads.
        unsafe {
            eval_raw(
                RawApiId::INT2E_SPH,
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
        .expect("per-quartet eval_raw");

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
                        "shls={shls:?} element={element} per-quartet={:.17e} batch={:.17e}",
                        single[element], actual[element]
                    );
                }
            }
        }
    }

    assert_eq!(
        single_mismatches, 0,
        "batched path must be BIT-identical to the per-quartet path; first: {first_report}"
    );
    assert_eq!(
        vendor_mismatches, 0,
        "batched path must match vendored libcint (max |diff| {max_vendor_diff:.3e}); \
         first: {first_report}"
    );
    assert!(
        max_vendor_diff < 1e-12,
        "max |diff| vs vendor {max_vendor_diff:.3e}"
    );
}

/// A batch whose class needs more Rys roots than the device kernel supports is
/// rejected as a whole — no partially-zeroed output.
#[test]
fn batch_rejects_classes_above_the_device_rys_ceiling() {
    let shell = BatchShell {
        l: 3,
        nprim: 1,
        nctr: 1,
        exponents: vec![0.9],
        coefficients: vec![1.0],
        center: [0.0, 0.0, 0.0],
    };
    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    // l-sum 12 -> nroots 7, above the device ceiling of 5.
    let result = evaluate_2e_quartet_batch(&backend, &[shell], &[[0, 0, 0, 0]]);
    assert!(
        result.is_err(),
        "an (f,f|f,f) batch must be rejected, not silently zeroed"
    );
}

/// Task 34-C — a device-resident basis must change the transfer accounting and
/// nothing else.
///
/// The acceptance bar from the plan is explicit: "`BatchStats.transfer_bytes`
/// for the second and later iterations drops to the descriptor table only;
/// results unchanged". Both halves are asserted here, because either one alone
/// is satisfiable by a bug — a residency that quietly re-uploads passes the
/// value check, and one that reads stale device memory passes the byte count.
#[test]
fn resident_basis_uploads_once_and_changes_nothing() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    let shells = batch_shells(&arrays);
    let list: Vec<[u32; 4]> = quartets
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    let reference = evaluate_2e_quartet_batch(&backend, &shells, &list).expect("throwaway batch");

    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("resident basis");
    assert_eq!(resident.reuse_count(), 0);

    let first = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("first");
    let second = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("second");
    let third = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("third");

    assert_eq!(resident.reuse_count(), 3);
    assert!(
        resident.upload_bytes() > 0,
        "the basis upload must be non-trivial or this test proves nothing"
    );

    // The first evaluation is charged the upload; later ones are not.
    assert_eq!(first.stats.basis_upload_bytes, resident.upload_bytes());
    assert_eq!(second.stats.basis_upload_bytes, 0);
    assert_eq!(third.stats.basis_upload_bytes, 0);
    assert_eq!(
        first.stats.transfer_bytes - resident.upload_bytes(),
        second.stats.transfer_bytes,
        "the residual transfer must be exactly the quartet tables"
    );
    assert_eq!(second.stats.transfer_bytes, third.stats.transfer_bytes);
    assert_eq!(
        second.stats.kernel_launch_count,
        first.stats.kernel_launch_count
    );

    // ... and every value is bit-identical across all four evaluations.
    for (label, run) in [("first", &first), ("second", &second), ("third", &third)] {
        assert_eq!(run.values.len(), reference.values.len(), "{label} length");
        let mismatches = run
            .values
            .iter()
            .zip(&reference.values)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "{label} resident evaluation must be BIT-identical to the throwaway-basis path"
        );
    }
}

/// Task 34-D — primitive-quartet screening at tolerance zero is the identity.
///
/// This is the same gate `cintx-driver` already enforces for Schwarz screening,
/// applied one level down: a screening bug that discards real work reads as a
/// speed-up, so the only defence is that the disabled configuration reproduces
/// the unscreened result bit for bit. The second half checks the knob is not
/// inert — a tolerance large enough to matter must actually change the answer,
/// or the identity above would prove nothing.
#[test]
fn primitive_screening_at_zero_is_the_identity() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    let shells = batch_shells(&arrays);
    let list: Vec<[u32; 4]> = quartets
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("resident basis");

    let exact = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("exact");
    let zero_tol = evaluate_2e_quartet_batch_with(
        &backend,
        &resident,
        &list,
        TwoEBatchOptions {
            primitive_tolerance: 0.0,
        },
    )
    .expect("tolerance-zero");

    let mismatches = exact
        .values
        .iter()
        .zip(&zero_tol.values)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        mismatches, 0,
        "primitive screening at tolerance 0 must be BIT-identical to no screening"
    );

    // A tolerance above every primitive quartet's scale factor must drop
    // everything; if it does not, the knob is not wired to the kernel at all.
    let everything_screened = evaluate_2e_quartet_batch_with(
        &backend,
        &resident,
        &list,
        TwoEBatchOptions {
            primitive_tolerance: f64::MAX,
        },
    )
    .expect("fully screened");
    assert!(
        everything_screened.values.iter().all(|v| *v == 0.0),
        "a tolerance above every scale factor must skip every primitive quartet"
    );
    assert!(
        exact.values.iter().any(|v| *v != 0.0),
        "the unscreened reference must be non-trivial"
    );
}
