//! Phase 26 (FND-03, D-07): manifest `complex_output` flag → `complex_interleaved`
//! → safe-API `Complex<f64>` round-trip proof.
//!
//! This is the FND-03 foundation proof: it verifies that a complex-output family
//! routed through `SessionRequest::evaluate` (the safe-API path) exposes its result
//! as a `Complex<f64>` view via `IntegralTensor::complex_values()`. The plumbing
//! under test is:
//!
//!   manifest `complex_output: bool` (Task 1)
//!     → planner `build_output_layout` sizes 2× and sets `complex_interleaved` (Task 2)
//!     → `IntegralTensor.complex_interleaved` (api.rs:375)
//!     → `complex_values()` returns `Some(Vec<Complex<f64>>)` (api.rs:603).
//!
//! SEAM (documented per Plan 26-01 Task 3): this plan does NOT register a GIAO
//! family kernel — `int1e_igovlp` and its kernel land in Cluster A (Plan 26-02).
//! To keep Plan 26-01 self-verifying WITHOUT 26-02, the round-trip assertion here
//! runs against the FIRST complex_output family already registered: the existing
//! `int1e_ovlp_spinor` family (backfilled `complex_output=true` in Task 1). It
//! proves the flag → `complex_interleaved` → `complex_values()` view path returns
//! `Some`. Plan 26-02 UPGRADES this test to the full D-07 assertion on
//! `int1e_igovlp` (imag non-zero somewhere; real part exactly 0.0) once the GIAO
//! kernel + host re=0 materialization land.
//!
//! Plain `#[test]` (NOT a vendor byte-identity parity test): this verifies the
//! safe-API complex round-trip plumbing, not numeric parity.

// Module gate widened to allow `--features rocm` (without cpu) per Phase 16-04 pattern.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

/// H2O STO-3G safe-API basis (clone of safe_api_arity2_parity.rs:173 + the Phase 22
/// non-zero gauge fixture geometry, `build_h2o_sto3g_common_orig`). The `rep`
/// argument selects the representation for every shell so the complex (spinor)
/// path can be exercised through the safe API.
fn build_h2o_sto3g_common_orig_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o = Atom::try_new(8, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atom_h2 =
        Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G exponents/coefficients (must match build_h2o_sto3g exactly).
    let shell_o1s = Arc::new(
        Shell::try_new(
            0,
            0,
            3,
            1,
            0,
            rep,
            arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
            arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
        )
        .unwrap(),
    );
    let shell_o2s = Arc::new(
        Shell::try_new(
            0,
            0,
            3,
            1,
            0,
            rep,
            arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
            arc_f64(&[-0.09996723, 0.39951283, 0.70011547]),
        )
        .unwrap(),
    );
    let shell_o2p = Arc::new(
        Shell::try_new(
            0,
            1,
            3,
            1,
            0,
            rep,
            arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
            arc_f64(&[0.15591627, 0.60768372, 0.39195739]),
        )
        .unwrap(),
    );
    let shell_h1_1s = Arc::new(
        Shell::try_new(
            1,
            0,
            3,
            1,
            0,
            rep,
            arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
            arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
        )
        .unwrap(),
    );
    let shell_h2_1s = Arc::new(
        Shell::try_new(
            2,
            0,
            3,
            1,
            0,
            rep,
            arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
            arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
        )
        .unwrap(),
    );

    let shells = vec![shell_o1s, shell_o2s, shell_o2p, shell_h1_1s, shell_h2_1s];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}

/// Drive a complex_output family through the safe API for one shell pair and
/// return its `IntegralTensor` complex view (or `None` if the flag did not flow).
fn complex_view_for_pair(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
    si: usize,
    sj: usize,
) -> Option<Vec<num_complex::Complex<f64>>> {
    let shell_tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
        .expect("ShellTuple construction must succeed for a valid 2-shell pair");
    let request = SessionRequest::new(
        operator_id,
        rep,
        basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request
        .query_workspace()
        .expect("query_workspace must succeed for a valid safe-API request");
    let output = query
        .evaluate()
        .expect("evaluate must succeed for a complex-output family");
    output.tensor.complex_values()
}

/// FND-03 round-trip: the manifest `complex_output` flag flows all the way to the
/// safe-API `Complex<f64>` view.
///
/// Target: `int1e_ovlp_spinor` (OperatorId 2) — the first registered complex_output
/// family (backfilled `complex_output=true` in Task 1). This proves the
/// flag → `complex_interleaved` → `complex_values()` path returns `Some`.
///
/// Plan 26-02 upgrades this to `int1e_igovlp` with the full D-07 imag/real
/// assertion once the GIAO kernel + host re=0 materialization land.
#[test]
fn giao_complex_roundtrip_flag_flows_to_complex_values_view() {
    // int1e_ovlp_spinor: OperatorId 2 (matches safe_api_arity2_parity.rs:822).
    let operator_id = OperatorId::new(2);
    let rep = Representation::Spinor;
    let (basis, shells) = build_h2o_sto3g_common_orig_safe_basis(rep);

    // (1) Flag → view path: a NON-SQUARE block (O-2p × O-1s, indices 2 × 0)
    // exercises the spinor path on a p×s block and confirms the safe-API
    // complex_values() view returns Some(...) — the core FND-03 round-trip.
    let complex_values = complex_view_for_pair(operator_id, rep, &basis, &shells, 2, 0);
    assert!(
        complex_values.is_some(),
        "manifest complex_output=true must flow through complex_interleaved so the \
         safe-API complex_values() view returns Some(...) — the FND-03 round-trip"
    );
    let values = complex_values.unwrap();
    assert!(
        !values.is_empty(),
        "complex_values() must return a non-empty Vec<Complex<f64>> for a complex family"
    );

    // (2) Non-zero-fill sanity across the assembled matrix: an individual
    // off-diagonal spinor block can be physically zero, but the full set of pairs
    // (which includes the non-zero self-overlap diagonal) must carry content —
    // this catches a silent zero-fill regression in the complex path. Every pair
    // also re-confirms complex_values() returns Some (the flag flows for all).
    let mut any_nonzero = false;
    for si in 0..shells.len() {
        for sj in 0..shells.len() {
            let view = complex_view_for_pair(operator_id, rep, &basis, &shells, si, sj)
                .expect("every spinor pair must expose a complex_values() view (flag flows)");
            if view
                .iter()
                .any(|c| c.re.abs() > 1e-18 || c.im.abs() > 1e-18)
            {
                any_nonzero = true;
            }
        }
    }
    assert!(
        any_nonzero,
        "the assembled complex spinor matrix must carry at least one non-zero \
         component (zero-fill regression?)"
    );
}

/// Full D-07 (Plan 26-02): drive `int1e_igovlp` (the registered Cluster-A GIAO
/// family) through the safe API and assert its `Complex<f64>` view is purely
/// imaginary — imaginary half NON-ZERO somewhere, real half EXACTLY 0.0.
///
/// `int1e_igovlp` is purely imaginary: the device emits real components, the host
/// materializes them as `[re=0, im=value]`. This is the canonical FND-03 proof on
/// an actual GIAO family (upgrades the Plan 26-01 seam which used int1e_ovlp_spinor).
#[test]
fn giao_igovlp_complex_view_is_purely_imaginary() {
    use cintx_ops::resolver::Resolver;

    // Resolve int1e_igovlp_cart's OperatorId BY SYMBOL (never a hardcoded integer —
    // the positional id is high since GIAO rows append at the manifest tail).
    let descriptor = Resolver::descriptor_by_symbol("int1e_igovlp_cart")
        .expect("int1e_igovlp_cart must be registered in the manifest");
    let operator_id = descriptor.id;
    let rep = Representation::Cart;
    let (basis, shells) = build_h2o_sto3g_common_orig_safe_basis(rep);

    // Cross-center NON-SQUARE pair (H1-1s shell 3 × O-2p shell 2): the igovlp gout
    // carries a c = ri - rj factor that vanishes same-center, so a cross-center
    // ket makes the integral genuinely non-zero. Non-square = D-12 transpose gate.
    let view = complex_view_for_pair(operator_id, rep, &basis, &shells, 3, 2)
        .expect("int1e_igovlp complex_output=true must expose a complex_values() view");
    assert!(!view.is_empty(), "complex_values() must be non-empty");

    let imag_nonzero = view.iter().any(|c| c.im.abs() > 1e-12);
    assert!(
        imag_nonzero,
        "int1e_igovlp: the imaginary half must be non-zero somewhere (D-07: GIAO \
         content landed, not silently zeroed)"
    );
    for (i, c) in view.iter().enumerate() {
        assert_eq!(
            c.re, 0.0,
            "int1e_igovlp: real part at element {i} must be EXACTLY 0.0 (D-07: \
             purely imaginary), got {}",
            c.re
        );
    }
}
