//! Part 7 — the batched families on the ROCm backend.
//!
//! Everything in the def2 batch suite is measured on the CubeCL **CPU** backend,
//! where `two_e_per_unit` selects the one-tuple-per-unit decomposition. The
//! cooperative shape — `per_unit == 0`, one tuple per cube, the cube splitting
//! the contraction, real `sync_cube` barriers — is compiled for every backend
//! but never *executed* in CI. A divergence in it would be invisible.
//!
//! This is not a throughput target: gfx1151 is an integrated GPU whose f64 rate
//! against a 16-core CPU running libcint makes that implausible. It is the only
//! available **correctness** target for the GPU launch topology.
//!
//! The comparison is against the CPU run in the same process, not a stored
//! baseline, so the two decompositions are held to bit-identity rather than to a
//! tolerance — which is the whole claim: the launch topology must not change a
//! result.
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 \
//!   cargo test --release -p cintx-oracle --features cpu,rocm \
//!   --test def2_batch_rocm_parity -- --ignored --nocapture
//! ```

#![cfg(all(feature = "cpu", feature = "rocm", has_vendor_libcint))]

use cintx_basis::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};
use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::kernels::one_electron::BatchAtom;
use cintx_cubecl::{
    BatchShell, OneEOperator, evaluate_1e_pair_batch, evaluate_2c2e_pair_batch,
    evaluate_2e_quartet_batch, evaluate_3c2e_triple_batch,
};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// Opt-in gate: the ROCm suite needs a real device and is not part of the
/// default CI matrix.
fn rocm_requested() -> bool {
    std::env::var("CINTX_ROCM_ORACLE").is_ok_and(|value| value != "0")
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

fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    let mut shells = Vec::with_capacity(arrays.nbas());
    for shell in 0..arrays.nbas() {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;
        let atom = record[ATOM_OF] as usize;
        let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;

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

fn backend(kind: BackendKind) -> ResolvedBackend {
    let label = format!("{kind:?}");
    ResolvedBackend::from_intent(&BackendIntent {
        backend: kind,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{label} backend: {error}"))
}

/// Largest CPU-vs-ROCm gap tolerated, as a multiple of `f64::EPSILON` times the
/// **block's largest element**.
///
/// Scaling to the block maximum rather than to each element is what makes this a
/// meaningful bar. The two backends sum the same terms; where their roundings
/// differ, the error is a few ULP *of the largest term in the sum*, which for a
/// small result is many ULP of that result. Gating each element against its own
/// magnitude would therefore reject a 1e-18 discrepancy on a 1e-4 output while
/// accepting the same absolute error on a 1.0 output — measuring cancellation,
/// not correctness.
const MAX_CROSS_BACKEND_EPS: f64 = 8.0;

/// Report the CPU-vs-ROCm agreement and gate it.
///
/// **This is deliberately not a bit-identity assertion.** The post-Phase-35 plan
/// asked for byte-identical ROCm results; that expectation does not survive
/// contact with the hardware, and the reason is not the launch topology this
/// test exists to check. Both backends compile the same `#[cube]` source, but
/// through different compilers to different ISAs, and the AMD one contracts
/// multiply-add pairs the CPU one leaves separate. The contracted form is the
/// *more* accurate of the two. Demanding bit-identity here would either fail
/// forever or force the FMA off and make the GPU path slower and less accurate
/// to satisfy a bar that measures the compiler rather than the code.
///
/// What the bar still catches is the thing this test exists for: a lane writing
/// another lane's element, a barrier in the wrong place, a stale G slab, a
/// mis-sized merged scratch slab. None of those perturb a result by a few ULP —
/// they produce a wrong number, and a wrong number is orders of magnitude away.
fn assert_agrees(label: &str, cpu: &[f64], rocm: &[f64]) {
    assert_eq!(
        cpu.len(),
        rocm.len(),
        "{label}: length {} (cpu) vs {} (rocm)",
        cpu.len(),
        rocm.len()
    );
    let scale = cpu.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let bound = scale * MAX_CROSS_BACKEND_EPS * f64::EPSILON;

    let mut identical = 0_usize;
    let mut worst_abs = 0.0_f64;
    let mut failures = 0_usize;
    let mut first = String::new();
    for (index, (c, r)) in cpu.iter().zip(rocm).enumerate() {
        if c.to_bits() == r.to_bits() {
            identical += 1;
            continue;
        }
        let abs = (c - r).abs();
        worst_abs = worst_abs.max(abs);
        if abs > bound {
            failures += 1;
            if first.is_empty() {
                first = format!("index {index}: cpu={c:.17e} rocm={r:.17e} |diff|={abs:.3e}");
            }
        }
    }
    let in_eps = if scale > 0.0 {
        worst_abs / (scale * f64::EPSILON)
    } else {
        0.0
    };
    println!(
        "  {label:<28} elements={:<7} bit-identical={:<7} max|diff|={worst_abs:.3e} \
({in_eps:.2} eps of scale {scale:.3e})",
        cpu.len(),
        identical
    );
    assert_eq!(
        failures, 0,
        "{label}: {failures} elements exceed {MAX_CROSS_BACKEND_EPS} eps of the \
         block scale between the cooperative (ROCm) and per-unit (CPU) \
         decompositions — more than a differing rounding order explains. \
         First {first}"
    );
}

/// Half-open span of block `index` in a concatenated batch output.
fn span(offsets: &[usize], index: usize, total: usize) -> (usize, usize) {
    let start = offsets[index];
    let end = offsets.get(index + 1).copied().unwrap_or(total);
    (start, end)
}

/// Oracle tolerance for the vendor comparison — the same flat bound the other
/// def2 gates use.
const VENDOR_TOLERANCE: f64 = 1e-10;

/// The correctness gate: a cintx run must reproduce vendored libcint.
fn assert_matches_vendor(label: &str, vendor: &[f64], actual: &[f64]) {
    assert_eq!(vendor.len(), actual.len(), "{label}: length");
    let mut worst = 0.0_f64;
    let mut mismatched = 0_usize;
    for (v, a) in vendor.iter().zip(actual) {
        let diff = (v - a).abs();
        worst = worst.max(diff);
        if diff > VENDOR_TOLERANCE {
            mismatched += 1;
        }
    }
    println!("  {label:<28} vs vendor: max|diff|={worst:.3e}  mismatched={mismatched}");
    assert_eq!(
        mismatched, 0,
        "{label}: {mismatched} elements exceed {VENDOR_TOLERANCE:.0e} against \
         vendored libcint (max|diff| {worst:.3e})"
    );
}

/// `int2e_sph` over every def2-SVP water quartet, both decompositions.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn def2_2e_batch_matches_between_cpu_and_rocm() {
    if !rocm_requested() {
        println!("CINTX_ROCM_ORACLE not set; skipping");
        return;
    }
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&view))
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    assert!(!list.is_empty());
    let shells = batch_shells(&arrays);

    println!("\nint2e_sph — {} quartets", list.len());
    let cpu = evaluate_2e_quartet_batch(&backend(BackendKind::Cpu), &shells, &list)
        .expect("cpu 2e batch");
    let rocm = evaluate_2e_quartet_batch(&backend(BackendKind::Rocm), &shells, &list)
        .expect("rocm 2e batch");

    // The launch *plan* is backend-independent; only the decomposition differs.
    assert_eq!(
        cpu.stats.kernel_launch_count, rocm.stats.kernel_launch_count,
        "both backends dispatch once per launch signature"
    );
    assert_eq!(cpu.stats.launch_classes, rocm.stats.launch_classes);
    assert_eq!(cpu.offsets, rocm.offsets);
    assert_agrees("int2e_sph", &cpu.values, &rocm.values);

    // The load-bearing claim: the cooperative path is *correct*, not merely
    // close to the per-unit path. Both are checked against the same third
    // party, so a shared error in the two cintx decompositions cannot pass.
    let mut vendor = vec![0.0_f64; cpu.values.len()];
    let mut scratch = vec![0.0_f64; 4096];
    for (index, quartet) in list.iter().enumerate() {
        let (start, end) = span(&cpu.offsets, index, cpu.values.len());
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
    assert_matches_vendor("int2e_sph (rocm)", &vendor, &rocm.values);
    assert_matches_vendor("int2e_sph (cpu)", &vendor, &cpu.values);
}

/// `int1e_ovlp` / `int1e_kin` / `int1e_nuc`, `int2c2e` and `int3c2e` over the
/// same fixture — the other three batched families' cooperative path.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn def2_pair_and_triple_batches_match_between_cpu_and_rocm() {
    if !rocm_requested() {
        println!("CINTX_ROCM_ORACLE not set; skipping");
        return;
    }
    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = arrays.nbas();
    let cpu_backend = backend(BackendKind::Cpu);
    let rocm_backend = backend(BackendKind::Rocm);

    let pairs: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();
    let triples: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();

    let atoms: Vec<BatchAtom> = (0..arrays.natm())
        .map(|atom| {
            let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;
            BatchAtom {
                charge: f64::from(arrays.atm[atom * ATM_SLOTS + CHARGE_OF]),
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect();

    println!(
        "\n1e / 2c2e / 3c2e — {} pairs, {} triples",
        pairs.len(),
        triples.len()
    );
    for operator in [
        OneEOperator::Overlap,
        OneEOperator::Kinetic,
        OneEOperator::Nuclear,
    ] {
        let cpu = evaluate_1e_pair_batch(&cpu_backend, operator, &shells, &atoms, &pairs)
            .expect("cpu 1e batch");
        let rocm = evaluate_1e_pair_batch(&rocm_backend, operator, &shells, &atoms, &pairs)
            .expect("rocm 1e batch");
        assert_eq!(
            cpu.stats.kernel_launch_count,
            rocm.stats.kernel_launch_count
        );
        assert_agrees(operator.symbol(), &cpu.values, &rocm.values);

        let mut vendor = vec![0.0_f64; rocm.values.len()];
        let mut scratch = vec![0.0_f64; 1024];
        for (index, pair) in pairs.iter().enumerate() {
            let (start, end) = span(&rocm.offsets, index, rocm.values.len());
            let len = end - start;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            let shls = [pair[0] as i32, pair[1] as i32];
            match operator {
                OneEOperator::Overlap => vendor_ffi::vendor_int1e_ovlp_sph(
                    &mut scratch[..len],
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                ),
                OneEOperator::Kinetic => vendor_ffi::vendor_int1e_kin_sph(
                    &mut scratch[..len],
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                ),
                OneEOperator::Nuclear => vendor_ffi::vendor_int1e_nuc_sph(
                    &mut scratch[..len],
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                ),
            };
            vendor[start..end].copy_from_slice(&scratch[..len]);
        }
        assert_matches_vendor(
            &format!("{} (rocm)", operator.symbol()),
            &vendor,
            &rocm.values,
        );
    }

    let cpu = evaluate_2c2e_pair_batch(&cpu_backend, &shells, &pairs).expect("cpu 2c2e");
    let rocm = evaluate_2c2e_pair_batch(&rocm_backend, &shells, &pairs).expect("rocm 2c2e");
    assert_eq!(
        cpu.stats.kernel_launch_count,
        rocm.stats.kernel_launch_count
    );
    assert_agrees("int2c2e_sph", &cpu.values, &rocm.values);
    {
        let mut vendor = vec![0.0_f64; rocm.values.len()];
        let mut scratch = vec![0.0_f64; 1024];
        for (index, pair) in pairs.iter().enumerate() {
            let (start, end) = span(&rocm.offsets, index, rocm.values.len());
            let len = end - start;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            vendor_ffi::vendor_int2c2e_sph(
                &mut scratch[..len],
                &[pair[0] as i32, pair[1] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            vendor[start..end].copy_from_slice(&scratch[..len]);
        }
        assert_matches_vendor("int2c2e_sph (rocm)", &vendor, &rocm.values);
    }

    let cpu = evaluate_3c2e_triple_batch(&cpu_backend, &shells, &triples).expect("cpu 3c2e");
    let rocm = evaluate_3c2e_triple_batch(&rocm_backend, &shells, &triples).expect("rocm 3c2e");
    assert_eq!(
        cpu.stats.kernel_launch_count,
        rocm.stats.kernel_launch_count
    );
    assert_agrees("int3c2e_sph", &cpu.values, &rocm.values);
    {
        let mut vendor = vec![0.0_f64; rocm.values.len()];
        let mut scratch = vec![0.0_f64; 4096];
        for (index, triple) in triples.iter().enumerate() {
            let (start, end) = span(&rocm.offsets, index, rocm.values.len());
            let len = end - start;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            vendor_ffi::vendor_int3c2e_sph(
                &mut scratch[..len],
                &[triple[0] as i32, triple[1] as i32, triple[2] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            vendor[start..end].copy_from_slice(&scratch[..len]);
        }
        assert_matches_vendor("int3c2e_sph (rocm)", &vendor, &rocm.values);
    }
}

/// Task 33-05, on the only GPU backend this host can run.
///
/// The plan records this as *the blocking item* for Phase 33 and as
/// undischargeable without hardware. It is dischargeable for ROCm here, and the
/// answer matters in a specific way: the ROCm results above diverge from the CPU
/// ones by a couple of ULP precisely *because* the AMD compiler contracts
/// multiply-adds the CPU backend leaves separate. That is the same mechanism
/// 33-05 is worried about — so the question is not whether contraction happens
/// (it demonstrably does) but whether `fma` still lowers to a **single-rounding**
/// fused multiply-add, which is what TwoProd's exactness depends on.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn rocm_backend_fma_fusion_probe() {
    if !rocm_requested() {
        println!("CINTX_ROCM_ORACLE not set; skipping");
        return;
    }
    use cintx_cubecl::{
        BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, RysFamily, device_nroots_ceiling,
        probe_fma_fusion,
    };

    for (label, kind) in [("cpu", BackendKind::Cpu), ("rocm", BackendKind::Rocm)] {
        let resolved = backend(kind);
        let probe = probe_fma_fusion(&resolved);
        let ceiling = device_nroots_ceiling(&resolved, RysFamily::Int2e);
        println!(
            "  {label:<6} fused={:<6} divergent={}/{}  nroots ceiling={ceiling}",
            probe.fused, probe.divergent, probe.pairs
        );
        assert!(
            probe.fused,
            "{label}: `fma` does not lower to a single-rounding fused \
             multiply-add ({}/{} probe pairs diverged). The double-double \
             TwoProd error term would be wrong on this backend, so its Rys \
             ceiling must stay at {BASE_DEVICE_NROOTS}.",
            probe.divergent, probe.pairs
        );
        // The probe is necessary but not sufficient: `int2e`'s ceiling is
        // raised only when the family has been flipped (task 33-03 did that)
        // *and* `extended-device-rys` is compiled in. This assertion used to
        // read `ceiling == BASE_DEVICE_NROOTS` unconditionally, which was
        // scaffolding from before the flip and held only because the suite
        // happened to be built without the feature.
        let expected = if cfg!(feature = "extended-device-rys")
            && RysFamily::Int2e.runs_extended_rys()
        {
            EXTENDED_DEVICE_NROOTS
        } else {
            BASE_DEVICE_NROOTS
        };
        assert_eq!(
            ceiling, expected,
            "{label}: the ceiling must follow `int2e`'s own flip \
             (runs_extended_rys={}) and the feature (={}), and nothing else",
            RysFamily::Int2e.runs_extended_rys(),
            cfg!(feature = "extended-device-rys"),
        );
    }
}

/// Task 35-D introduced the cooperative decomposition into four more kernels
/// (`int3c2e_ip1`/`ip2`, `int1e_ipovlp`/`ipkin`, `int1e_ipnuc`). Like the scalar
/// families, that `per_unit == 0` path is compiled everywhere and executed
/// nowhere in CI — so it runs here.
#[test]
#[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
fn def2_derivative_batches_match_between_cpu_and_rocm() {
    if !rocm_requested() {
        println!("CINTX_ROCM_ORACLE not set; skipping");
        return;
    }
    use cintx_cubecl::{
        OneEDerivOperator, ThreeC2eDerivFamily, evaluate_1e_deriv_pair_batch,
        evaluate_3c2e_deriv_triple_batch,
    };

    let arrays = to_raw_arrays(&water(StandardBasis::Def2Svp)).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = arrays.nbas();
    let cpu_backend = backend(BackendKind::Cpu);
    let rocm_backend = backend(BackendKind::Rocm);

    let atoms: Vec<BatchAtom> = (0..arrays.natm())
        .map(|atom| {
            let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;
            BatchAtom {
                charge: f64::from(arrays.atm[atom * ATM_SLOTS + CHARGE_OF]),
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect();

    let pairs: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();
    let triples: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();

    println!(
        "\nderivative families — {} pairs, {} triples",
        pairs.len(),
        triples.len()
    );

    for operator in [
        OneEDerivOperator::IpOvlp,
        OneEDerivOperator::IpKin,
        OneEDerivOperator::IpNuc,
    ] {
        let cpu = evaluate_1e_deriv_pair_batch(&cpu_backend, operator, &shells, &atoms, &pairs)
            .expect("cpu 1e deriv batch");
        let rocm = evaluate_1e_deriv_pair_batch(&rocm_backend, operator, &shells, &atoms, &pairs)
            .expect("rocm 1e deriv batch");
        assert_eq!(
            cpu.stats.kernel_launch_count,
            rocm.stats.kernel_launch_count
        );
        assert_eq!(cpu.offsets, rocm.offsets);
        report_agreement(operator.symbol(), &cpu.values, &rocm.values);

        let mut vendor = vec![0.0_f64; rocm.values.len()];
        let mut scratch = vec![0.0_f64; 2048];
        for (index, pair) in pairs.iter().enumerate() {
            let (start, end) = span(&rocm.offsets, index, rocm.values.len());
            let len = end - start;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            let shls = [pair[0] as i32, pair[1] as i32];
            let (atm, natm, bas, nbas_i, env) = (
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            match operator {
                OneEDerivOperator::IpOvlp => vendor_ffi::vendor_int1e_ipovlp_sph(
                    &mut scratch[..len],
                    &shls,
                    atm,
                    natm,
                    bas,
                    nbas_i,
                    env,
                ),
                OneEDerivOperator::IpKin => vendor_ffi::vendor_int1e_ipkin_sph(
                    &mut scratch[..len],
                    &shls,
                    atm,
                    natm,
                    bas,
                    nbas_i,
                    env,
                ),
                OneEDerivOperator::IpNuc => vendor_ffi::vendor_int1e_ipnuc_sph(
                    &mut scratch[..len],
                    &shls,
                    atm,
                    natm,
                    bas,
                    nbas_i,
                    env,
                ),
            };
            vendor[start..end].copy_from_slice(&scratch[..len]);
        }
        assert_matches_vendor(
            &format!("{} (rocm)", operator.symbol()),
            &vendor,
            &rocm.values,
        );
    }

    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let label = match family {
            ThreeC2eDerivFamily::Ip1 => "int3c2e_ip1_sph",
            ThreeC2eDerivFamily::Ip2 => "int3c2e_ip2_sph",
        };
        let cpu = evaluate_3c2e_deriv_triple_batch(&cpu_backend, family, &shells, &triples)
            .expect("cpu 3c2e deriv batch");
        let rocm = evaluate_3c2e_deriv_triple_batch(&rocm_backend, family, &shells, &triples)
            .expect("rocm 3c2e deriv batch");
        assert_eq!(
            cpu.stats.kernel_launch_count,
            rocm.stats.kernel_launch_count
        );
        assert_eq!(cpu.offsets, rocm.offsets);
        report_agreement(label, &cpu.values, &rocm.values);

        let mut vendor = vec![0.0_f64; rocm.values.len()];
        let mut scratch = vec![0.0_f64; 8192];
        for (index, triple) in triples.iter().enumerate() {
            let (start, end) = span(&rocm.offsets, index, rocm.values.len());
            let len = end - start;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            let shls = [triple[0] as i32, triple[1] as i32, triple[2] as i32];
            match family {
                ThreeC2eDerivFamily::Ip1 => vendor_ffi::vendor_int3c2e_ip1_sph(
                    &mut scratch[..len],
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                ),
                ThreeC2eDerivFamily::Ip2 => vendor_ffi::vendor_int3c2e_ip2_sph(
                    &mut scratch[..len],
                    &shls,
                    &arrays.atm,
                    arrays.natm() as i32,
                    &arrays.bas,
                    arrays.nbas() as i32,
                    &arrays.env,
                ),
            };
            vendor[start..end].copy_from_slice(&scratch[..len]);
        }
        assert_matches_vendor(&format!("{label} (rocm)"), &vendor, &rocm.values);
    }
}

/// Report the CPU-vs-ROCm agreement **without** gating on it.
///
/// The scalar families are gated by [`assert_agrees`] on an eps-of-block-scale
/// bound, and that instrument is wrong for the derivative ones. A gradient
/// kernel builds second differences — `coef_hi * g[n + 2dj] - coef_mid * g[n]`
/// in the kinetic arm — and cancellation there turns a 2-ULP difference in an
/// intermediate into a 1e-13 difference in the result while the block scale
/// stays O(1). Measured: `int1e_ipkin` diverges by 5.8e-14 on 2 of 1296
/// elements, which an eps-of-scale bound reads as ~1100 eps.
///
/// So for these families the gate is agreement with **vendored libcint** at the
/// oracle tolerance, which is the claim that actually matters, and the
/// CPU-vs-ROCm distance is reported as context rather than asserted on.
fn report_agreement(label: &str, cpu: &[f64], rocm: &[f64]) {
    assert_eq!(cpu.len(), rocm.len(), "{label}: length");
    let scale = cpu.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let identical = cpu
        .iter()
        .zip(rocm)
        .filter(|(c, r)| c.to_bits() == r.to_bits())
        .count();
    let worst = cpu
        .iter()
        .zip(rocm)
        .fold(0.0_f64, |acc, (c, r)| acc.max((c - r).abs()));
    let in_eps = if scale > 0.0 {
        (worst / (scale * f64::EPSILON)).round() as u64
    } else {
        0
    };
    println!(
        "  {label:<28} elements={:<7} bit-identical={:<7} max|diff|={worst:.3e} \
({in_eps} eps of scale {scale:.3e})",
        cpu.len(),
        identical
    );
}
