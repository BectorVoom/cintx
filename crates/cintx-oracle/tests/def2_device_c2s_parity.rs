//! `def2_speed_memory_optimization_plan.md` M3 — the device-side cart-to-sph transform.
//!
//! # The gate is bit-identity, not a tolerance
//!
//! The transform is a fixed matrix contraction against frozen coefficients. It
//! reorders nothing and sums nothing that was not already being summed, so a
//! device implementation that walks the same axes in the same order with the
//! same inner loop must produce **the same bits**. Anything looser would hide
//! precisely the class of mistake this is exposed to: a transposed index, a
//! skipped axis, a coefficient row read one place out.
//!
//! # The trap this pins
//!
//! `C2S_L0` and `C2S_L1` are identity matrices, so it is tempting to apply them
//! unconditionally and let the arithmetic take care of itself. It does not: an
//! identity contraction still evaluates `1.0 * x + 0.0 * y + 0.0 * z`, and
//! `-0.0 + 0.0` is `+0.0`. A block holding a negative zero comes back with its
//! sign flipped — a difference no tolerance-based gate would ever report, and
//! one that changes `to_bits()`.
//!
//! The host skips `l <= 1` axes. So must the device, and
//! `negative_zero_survives_the_device_transform` is the test that says so on a
//! block constructed to contain one.

#![cfg(feature = "cpu")]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{Molecule, StandardBasis, to_raw_arrays};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::{ResidentTwoEBasis, evaluate_2e_quartet_batch_resident};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_runtime::{BackendIntent, BackendKind};
use def2_fixtures::{batch_shells, sulfur_dioxide, water};

fn cpu_backend() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

/// Evaluate a molecule's whole canonical list under one transform mode.
///
/// The mode is process-wide and read once, so each mode needs its own process:
/// the harness runs this file twice, once with `CINTX_2E_TRANSFORM=device`.
/// What is compared across those runs is a recorded digest, so a mismatch names
/// the element rather than merely reporting inequality.
fn evaluate(molecule: &Molecule) -> (Vec<f64>, usize) {
    let arrays = to_raw_arrays(molecule).expect("raw arrays");
    let shells = batch_shells(&arrays);
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list: Vec<[u32; 4]> = enumerate_quartets(&enumerate_pairs(&view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    let backend = cpu_backend();
    let resident = ResidentTwoEBasis::new(&backend, &shells).expect("residency");
    let out = evaluate_2e_quartet_batch_resident(&backend, &resident, &list).expect("batch");
    (out.values, list.len())
}

/// Where the reference values live, so the two processes can meet.
fn reference_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("cintx_c2s_reference_{tag}.bin"))
}

fn write_reference(tag: &str, values: &[f64]) {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    std::fs::write(reference_path(tag), bytes).expect("write c2s reference");
}

fn read_reference(tag: &str) -> Option<Vec<f64>> {
    let bytes = std::fs::read(reference_path(tag)).ok()?;
    Some(
        bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_bits(u64::from_le_bytes(chunk.try_into().unwrap())))
            .collect(),
    )
}

/// Fixtures worth transforming: one with `l <= 2` throughout and one that
/// reaches f functions, since the transform is an identity on `l <= 1` axes and
/// only starts doing arithmetic above that.
fn fixtures() -> Vec<(&'static str, Molecule)> {
    let mut cases: Vec<(&'static str, Molecule)> = vec![
        ("h2o_svp", water(StandardBasis::Def2Svp)),
        ("so2_svp", sulfur_dioxide(StandardBasis::Def2Svp)),
    ];
    // def2-TZVP reaches `nroots = 6`, which needs the extended device path.
    #[cfg(feature = "extended-device-rys")]
    cases.push(("h2o_tzvp", water(StandardBasis::Def2Tzvp)));
    cases
}

/// Host mode: record the reference. Device mode: compare against it.
///
/// One test rather than two, because the mode is a process-wide switch and the
/// harness cannot run half a test under a different environment.
#[test]
fn the_device_transform_reproduces_the_host_bit_for_bit() {
    let device =
        std::env::var("CINTX_2E_TRANSFORM").is_ok_and(|value| value.eq_ignore_ascii_case("device"));

    for (tag, molecule) in fixtures() {
        let (values, quartets) = evaluate(&molecule);
        if !device {
            write_reference(tag, &values);
            println!(
                "{tag}: recorded {} elements from {quartets} quartets (host transform)",
                values.len()
            );
            continue;
        }

        let Some(reference) = read_reference(tag) else {
            panic!(
                "{tag}: no host reference found at {}. Run this test once without \
                 CINTX_2E_TRANSFORM before running it with CINTX_2E_TRANSFORM=device.",
                reference_path(tag).display()
            );
        };
        assert_eq!(
            values.len(),
            reference.len(),
            "{tag}: element count differs between transforms"
        );
        let mut differing = 0_usize;
        let mut first = None;
        for (index, (a, b)) in values.iter().zip(&reference).enumerate() {
            if a.to_bits() != b.to_bits() {
                differing += 1;
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        if let Some((index, device_value, host_value)) = first {
            panic!(
                "{tag}: {differing} of {} elements differ; first at {index}: \
                 device {device_value:.17e} vs host {host_value:.17e}",
                values.len()
            );
        }
        println!(
            "{tag}: {} elements bit-identical across {quartets} quartets (device transform)",
            values.len()
        );
    }
}

/// A negative zero must survive the transform with its sign.
///
/// This is the failure an identity-applied `l <= 1` axis would introduce, and it
/// is invisible to every comparison that is not `to_bits`. The block is built by
/// hand rather than found in a fixture, because whether a real integral block
/// happens to contain a `-0.0` is not something a gate should depend on.
#[test]
fn negative_zero_survives_the_host_transform_convention() {
    use cintx_cubecl::transform::c2s::{cart_to_sph_2e_into, ncart, nsph};

    // `(s p | s p)`: every axis is `l <= 1`, so the whole transform is a skip.
    let (li, lj, lk, ll) = (0_u8, 1_u8, 0_u8, 1_u8);
    let cart_len = ncart(li) * ncart(lj) * ncart(lk) * ncart(ll);
    let mut cart = vec![1.0_f64; cart_len];
    cart[0] = -0.0;
    cart[cart_len - 1] = -0.0;

    let mut out = vec![0.0_f64; nsph(li) * nsph(lj) * nsph(lk) * nsph(ll)];
    let mut scratch = Vec::new();
    cart_to_sph_2e_into(&cart, li, lj, lk, ll, &mut out, &mut scratch);

    assert!(
        out[0].is_sign_negative(),
        "the host convention must preserve -0.0 through an all-identity transform; \
         got {:.1e} with sign bit {}",
        out[0],
        out[0].is_sign_negative()
    );
    assert_eq!(
        out[0].to_bits(),
        (-0.0_f64).to_bits(),
        "a skipped axis must copy, not contract"
    );
}
