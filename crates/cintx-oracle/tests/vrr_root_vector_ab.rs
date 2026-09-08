//! The root-vector VRR (`math::root_vec`, plan §12/§13) must change no bits.
//!
//! `vrr_fill_axis_roots` and `vrr_fill_axis_roots_ket` replace the per-root
//! scalar recurrence in four device kernels with one `Vector`-lane chain. Every
//! operation is elementwise on the same operands in the same order and no fused
//! multiply-add is introduced, so the claim is not "close enough" — it is that
//! the output is bit-for-bit what it was. This file is the end-to-end gate for
//! that claim on the 2c2e and 3c2e families, the way `gth_profile`'s
//! `CINTX_GTH_DUMP` / `CINTX_GTH_COMPARE` pair is for `two_electron`.
//!
//! Default run: vendor parity for all four families over a def2-TZVP water,
//! which reaches `nroots` 3–5 — higher angular momentum than the STO-3G sweeps
//! in `center_2c2e_parity` / `center_3c2e_parity`, and therefore the widths
//! (3 and 5 especially) where a vector lane count could go wrong.
//!
//! Bit-level A/B:
//!
//! ```text
//! CINTX_VRR_DUMP=<dir>    cargo test --release -p cintx-oracle --features cpu \
//!                              --test vrr_root_vector_ab
//! # flip the comptime `nroots > 1` arms off, rebuild, then:
//! CINTX_VRR_COMPARE=<dir> cargo test --release -p cintx-oracle --features cpu \
//!                              --test vrr_root_vector_ab
//! ```

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{
    ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP, RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{
    BatchShell, ThreeC2eDerivFamily, evaluate_2c2e_pair_batch, evaluate_3c2e_deriv_triple_batch,
    evaluate_3c2e_triple_batch,
};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// The unified 2c2e / 3c2e tolerance (D-15).
const TOLERANCE: f64 = 1e-12;

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

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Which families this gate sweeps, and how many components each returns.
#[derive(Clone, Copy)]
enum Family {
    /// `center_2c2e_kernel` — the ket-raising VRR, two shells.
    Int2c2e,
    /// `center_3c2e_scalar_kernel` — the ket-raising VRR, three shells.
    Int3c2e,
    /// `center_3c2e_ip1_kernel` — the bra-raising VRR, rank 3.
    Int3c2eIp1,
    /// `center_3c2e_ip2_kernel` — the bra-raising VRR, rank 3.
    Int3c2eIp2,
}

impl Family {
    fn label(self) -> &'static str {
        match self {
            Family::Int2c2e => "int2c2e_sph",
            Family::Int3c2e => "int3c2e_sph",
            Family::Int3c2eIp1 => "int3c2e_ip1_sph",
            Family::Int3c2eIp2 => "int3c2e_ip2_sph",
        }
    }

    fn api_id(self) -> RawApiId {
        match self {
            Family::Int2c2e => RawApiId::INT2C2E_SPH,
            Family::Int3c2e => RawApiId::INT3C2E_SPH,
            Family::Int3c2eIp1 => RawApiId::INT3C2E_IP1_SPH,
            Family::Int3c2eIp2 => RawApiId::INT3C2E_IP2_SPH,
        }
    }

    fn ncomp(self) -> usize {
        match self {
            Family::Int2c2e | Family::Int3c2e => 1,
            Family::Int3c2eIp1 | Family::Int3c2eIp2 => 3,
        }
    }

    fn nshells(self) -> usize {
        match self {
            Family::Int2c2e => 2,
            _ => 3,
        }
    }

    fn vendor(
        self,
        out: &mut [f64],
        shls: &[i32],
        atm: &[i32],
        natm: i32,
        bas: &[i32],
        nbas: i32,
        env: &[f64],
    ) {
        match self {
            Family::Int2c2e => {
                let pair: [i32; 2] = [shls[0], shls[1]];
                vendor_ffi::vendor_int2c2e_sph(out, &pair, atm, natm, bas, nbas, env)
            }
            Family::Int3c2e => {
                let triple: [i32; 3] = [shls[0], shls[1], shls[2]];
                vendor_ffi::vendor_int3c2e_sph(out, &triple, atm, natm, bas, nbas, env)
            }
            Family::Int3c2eIp1 => {
                let triple: [i32; 3] = [shls[0], shls[1], shls[2]];
                vendor_ffi::vendor_int3c2e_ip1_sph(out, &triple, atm, natm, bas, nbas, env)
            }
            Family::Int3c2eIp2 => {
                let triple: [i32; 3] = [shls[0], shls[1], shls[2]];
                vendor_ffi::vendor_int3c2e_ip2_sph(out, &triple, atm, natm, bas, nbas, env)
            }
        };
    }
}

/// Sweep every shell tuple of `family`, checking cintx against the vendor and
/// returning cintx's values concatenated in sweep order.
fn sweep(family: Family, arrays: &RawArrays) -> Vec<f64> {
    let atm = &arrays.atm;
    let bas = &arrays.bas;
    let env = &arrays.env;
    let natm = arrays.natm() as i32;
    let nbas = arrays.nbas() as i32;
    let n_shells = arrays.nbas();

    let shell_nsph: Vec<usize> = (0..n_shells)
        .map(|s| nsph_for_l(bas[s * BAS_SLOTS + ANG_OF]))
        .collect();

    let mut values = Vec::new();
    let mut worst = 0.0_f64;
    let mut any_nonzero = false;
    let mut refused = 0usize;

    // The third index is fixed to a single auxiliary-slot shell for the 3-index
    // families so the sweep stays quadratic rather than cubic; every `(l_i, l_j)`
    // pair — and so every `nroots` the family reaches — is still covered.
    let k_shells: Vec<usize> = match family.nshells() {
        2 => vec![usize::MAX],
        _ => (0..n_shells).step_by(n_shells.div_ceil(4).max(1)).collect(),
    };

    for i_sh in 0..n_shells {
        for j_sh in 0..n_shells {
            for &k_sh in &k_shells {
                let (shls, n_elem) = if family.nshells() == 2 {
                    (
                        vec![i_sh as i32, j_sh as i32],
                        shell_nsph[i_sh] * shell_nsph[j_sh],
                    )
                } else {
                    (
                        vec![i_sh as i32, j_sh as i32, k_sh as i32],
                        shell_nsph[i_sh] * shell_nsph[j_sh] * shell_nsph[k_sh],
                    )
                };
                let n_elem = n_elem * family.ncomp();

                let mut vendor_out = vec![0.0_f64; n_elem];
                let mut cintx_out = vec![0.0_f64; n_elem];

                family.vendor(&mut vendor_out, &shls, atm, natm, bas, nbas, env);
                let evaluated = unsafe {
                    eval_raw(
                        family.api_id(),
                        Some(&mut cintx_out),
                        None,
                        &shls,
                        atm,
                        bas,
                        env,
                        None,
                        None,
                    )
                };
                // The derivative families raise the bra by one, which can push
                // `nroots` past the device ceiling; that refusal is the
                // family's documented fail-closed contract (FND-02), not a
                // result to compare. Skip the tuple and record it.
                if let Err(e) = evaluated {
                    let message = format!("{e:?}");
                    assert!(
                        message.contains("unsupported_nrys_roots"),
                        "{}: eval_raw failed at {shls:?}: {message}",
                        family.label()
                    );
                    refused += 1;
                    continue;
                }

                for (v, c) in vendor_out.iter().zip(&cintx_out) {
                    if v.abs() > 1e-18 {
                        any_nonzero = true;
                    }
                    worst = worst.max((v - c).abs());
                }
                values.extend_from_slice(&cintx_out);
            }
        }
    }

    assert!(any_nonzero, "{}: sweep produced only zeros", family.label());
    assert!(
        worst < TOLERANCE,
        "{}: max|diff| vs vendor {worst:.3e} exceeds {TOLERANCE:.0e}",
        family.label()
    );
    println!(
        "  {:<16} {:>9} values   max|diff| {:.3e}   {refused} above the device Rys ceiling",
        family.label(),
        values.len(),
        worst
    );
    values
}

fn sanitize(label: &str) -> String {
    label.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
}

#[test]
fn root_vector_vrr_matches_vendor_and_is_bit_stable() {
    // def2-TZVP reaches d and f shells, so the 2c2e and 3c2e classes here span
    // `nroots` 1..5 — the odd widths 3 and 5 included, which is the point.
    let molecule = water(StandardBasis::Def2Tzvp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");

    let dump = std::env::var("CINTX_VRR_DUMP").ok();
    let compare = std::env::var("CINTX_VRR_COMPARE").ok();

    for family in [
        Family::Int2c2e,
        Family::Int3c2e,
        Family::Int3c2eIp1,
        Family::Int3c2eIp2,
    ] {
        let values = sweep(family, &arrays);

        if let Some(dir) = &dump {
            let path = std::path::Path::new(dir).join(format!("{}.f64", sanitize(family.label())));
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).expect("dump");
            println!("  dumped {} values to {}", values.len(), path.display());
        }
        if let Some(dir) = &compare {
            let path = std::path::Path::new(dir).join(format!("{}.f64", sanitize(family.label())));
            let bytes = std::fs::read(&path).expect("reference dump");
            let reference: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(
                reference.len(),
                values.len(),
                "{}: dump length",
                family.label()
            );
            let differing = reference
                .iter()
                .zip(&values)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "{}: {differing} of {} values differ in bits from the reference dump",
                family.label(),
                values.len()
            );
            println!("  {} bit-identical to the dump", family.label());
        }
    }
}

/// The batch shell table, from the raw `bas`/`env` arrays. The libcint env
/// coefficient block is column-major; `BatchShell::coefficients` is
/// primitive-major (WR-03).
fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    let mut shells = Vec::with_capacity(arrays.nbas());
    for shell in 0..arrays.nbas() {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;
        let coord_ptr = arrays.atm
            [record[cintx_compat::raw::ATOM_OF] as usize * cintx_compat::raw::ATM_SLOTS + PTR_COORD]
            as usize;
        let exponents = arrays.env[exp_ptr..exp_ptr + nprim].to_vec();
        let mut coefficients = Vec::with_capacity(nprim * nctr);
        for p in 0..nprim {
            for c in 0..nctr {
                coefficients.push(arrays.env[coeff_ptr + c * nprim + p]);
            }
        }
        shells.push(BatchShell {
            l: record[ANG_OF] as u8,
            nprim: nprim as u32,
            nctr: nctr as u32,
            exponents,
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

/// Batched-dispatch timing for the two 3c2e derivative kernels, on a workload
/// large enough that the dispatch dominates the launch overhead.
///
/// `cargo test --release -p cintx-oracle --features cpu --test vrr_root_vector_ab \
///      -- --ignored --nocapture`
#[test]
#[ignore = "timing probe; run explicitly in release"]
fn root_vector_vrr_throughput() {
    // SO2 at def2-TZVP: 45 shells through d, so the classes span `nroots` 2..5
    // — the widths the vector VRR is selected at.
    let molecule = Molecule::new(
        vec![
            AtomSpec::from_angstrom("S", [0.0, 0.0, 0.0]).unwrap(),
            AtomSpec::from_angstrom("O", [0.0, 1.238, 0.723]).unwrap(),
            AtomSpec::from_angstrom("O", [0.0, -1.238, 0.723]).unwrap(),
        ],
        StandardBasis::Def2Tzvp,
    );
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let nbas = arrays.nbas();
    // The full triple list is cubic; step the auxiliary index so the run stays
    // seconds rather than minutes while every class is still represented.
    // def2-TZVP carries f shells, and the derivative headroom raises one index
    // by a unit — `l` sums past 7 reach `nroots = 6`, which the device refuses
    // without `extended-device-rys` (FND-02). Keep the classes it evaluates.
    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| {
                (0..nbas)
                    .step_by(2)
                    .map(move |k| [i as u32, j as u32, k as u32])
            })
        })
        .filter(|t| {
            let l_sum: u32 = t.iter().map(|&s| shells[s as usize].l as u32).sum();
            l_sum + 1 <= 7
        })
        .collect();
    println!("SO2 / def2-TZVP: {nbas} shells, {} triples", list.len());

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend");

    // The base families first: `center_2c2e_kernel` and
    // `center_3c2e_scalar_kernel`, the two ket-raising sites.
    let pairs: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();
    let _ = evaluate_2c2e_pair_batch(&backend, &shells, &pairs).expect("2c2e batch");
    let mut best = f64::INFINITY;
    for _round in 0..7 {
        let start = std::time::Instant::now();
        let out = evaluate_2c2e_pair_batch(&backend, &shells, &pairs).expect("2c2e batch");
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        assert!(out.values.iter().any(|v| v.abs() > 1e-18));
    }
    println!(
        "  {:<16} best {best:8.1} ms  ({} pairs)",
        "int2c2e_sph",
        pairs.len()
    );

    let base_list: Vec<[u32; 3]> = list
        .iter()
        .copied()
        .filter(|t| {
            let l_sum: u32 = t.iter().map(|&s| shells[s as usize].l as u32).sum();
            l_sum <= 7
        })
        .collect();
    let _ = evaluate_3c2e_triple_batch(&backend, &shells, &base_list).expect("3c2e batch");
    let mut best = f64::INFINITY;
    for _round in 0..7 {
        let start = std::time::Instant::now();
        let out = evaluate_3c2e_triple_batch(&backend, &shells, &base_list).expect("3c2e batch");
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        assert!(out.values.iter().any(|v| v.abs() > 1e-18));
    }
    println!(
        "  {:<16} best {best:8.1} ms  ({} triples)",
        "int3c2e_sph",
        base_list.len()
    );

    for family in [ThreeC2eDerivFamily::Ip1, ThreeC2eDerivFamily::Ip2] {
        let label = match family {
            ThreeC2eDerivFamily::Ip1 => "int3c2e_ip1_sph",
            ThreeC2eDerivFamily::Ip2 => "int3c2e_ip2_sph",
        };
        // One untimed run so the JIT is not in the measurement.
        let _ = evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &list)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut best = f64::INFINITY;
        for _round in 0..7 {
            let start = std::time::Instant::now();
            let out = evaluate_3c2e_deriv_triple_batch(&backend, family, &shells, &list)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
            assert!(out.values.iter().any(|v| v.abs() > 1e-18));
        }
        println!("  {label:<16} best {best:8.1} ms");
    }
}
