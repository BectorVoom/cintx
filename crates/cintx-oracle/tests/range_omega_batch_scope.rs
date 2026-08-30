//! D-PBC-24 P0 — the BATCH surfaces must not silently drop `range_omega`.
//!
//! Stages 0–3 taught the scalar `SessionRequest` path to honour
//! `ExecutionOptions::range_omega`. The batched device kernels were never
//! taught: `two_electron`, `center_3c2e` and `center_2c2e`'s batch launchers
//! evaluate the full-range Coulomb operator unconditionally. Four entry points
//! held an `ExecutionOptions` and never read ω out of it, so a caller who set
//! `.with_range_omega(-0.8)` and then batched got the **full-range** integrals
//! back with no error — the one outcome the design forbids in writing: *"a
//! full-range substitute must never ship: it runs, converges, and is silently a
//! different method."*
//!
//! Three of the four fail closed, naming the scalar route that does honour ω:
//!
//! * `PairBatchRequest` and `TripleBatchRequest`, through the shared
//!   `check_batch_request_scope`;
//! * `QuartetBatchRequest`, now folded onto that same helper rather than
//!   carrying its own copy of the checks.
//!
//! The fourth, the `(s,s|s,s)` `int2e_cart` pilot inside `BatchRequest`, is not
//! a user-facing refusal point: it is an internal fast path with a correct
//! fallback. It declines itself under a set ω and the batch completes on the
//! scalar route, which reads ω. That is asserted numerically here — a fallback
//! that returned full-range numbers would be the same silent substitution by
//! another name.
//!
//! No vendor build needed; `range_omega_parity.rs` is the numeric gate against
//! libcint.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_ops::resolver::Resolver;
use cintx_rs::prelude::{
    BatchRequest, FacadeError, PairBatchRequest, QuartetBatchRequest, SessionRequest,
    TripleBatchRequest,
};
use cintx_runtime::{BackendIntent, BackendKind, ExecutionOptions};
use std::sync::Arc;

/// Four s-shells on four centres. `s` keeps every surface in scope: the
/// `(s,s|s,s)` pilot admits only uncontracted single-primitive s shells, and
/// `rys_order = 1` for every tuple, so short range doubles to 2 roots and stays
/// well inside the base device ceiling — the refusals under test are about ω,
/// not about root counts.
const EXPS: [f64; 4] = [0.9, 0.6, 1.3, 0.75];
const COORDS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.3, 0.0, 1.7],
    [1.1, 0.4, 0.0],
    [0.0, 1.6, 0.9],
];

/// Non-zero ω on both sides of libcint's sign convention: `> 0` long range
/// (`erf`), `< 0` short range (`erfc`).
const SEPARATED: [f64; 2] = [0.8, -0.8];

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn s_basis(representation: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atoms: Vec<Atom> = COORDS
        .iter()
        .map(|&c| Atom::try_new(2, c, NuclearModel::Point, None, None).unwrap())
        .collect();
    let atoms = Arc::from(atoms.into_boxed_slice());

    let shells: Vec<Arc<Shell>> = (0..4)
        .map(|s| {
            Arc::new(
                Shell::try_new(
                    s as u32,
                    0,
                    1,
                    1,
                    0,
                    representation,
                    arc_f64(&[EXPS[s]]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            )
        })
        .collect();

    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}

fn cpu_options(range_omega: Option<f64>) -> ExecutionOptions {
    ExecutionOptions {
        backend_intent: BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        range_omega,
        ..Default::default()
    }
}

fn operator(symbol: &str) -> OperatorId {
    Resolver::descriptor_by_symbol(symbol)
        .unwrap_or_else(|_| panic!("{symbol} is a registered operator"))
        .id
}

/// A refusal is only useful if it says what it refused. Every ω rejection has
/// to name `range_omega`, so a caller reading the message knows the parameter
/// was seen and declined rather than never plumbed.
fn assert_names_range_omega(error: &FacadeError, surface: &str) {
    assert!(
        matches!(error, FacadeError::UnsupportedApi { .. }),
        "{surface}: a set range_omega must be UnsupportedApi, got {error:?}"
    );
    let text = format!("{error}");
    assert!(
        text.contains("range_omega"),
        "{surface}: the refusal must name range_omega, got: {text}"
    );
}

/// `TripleBatchRequest` is `int3c2e` — the aux_e2 half of every RS density
/// fitting route, and so the first thing a range-separated periodic driver
/// would hit.
#[test]
fn triple_batch_refuses_a_set_range_omega() {
    let (basis, _) = s_basis(Representation::Spheric);
    for omega in SEPARATED {
        let error = TripleBatchRequest::new(
            operator("int3c2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1, 2]],
            cpu_options(Some(omega)),
        )
        .evaluate()
        .expect_err("a set range_omega must be refused by the triple batch surface");
        assert_names_range_omega(&error, "triple-batch");
    }
}

/// `PairBatchRequest` is `int2c2e` — the other half of aux_e2.
#[test]
fn pair_batch_refuses_a_set_range_omega() {
    let (basis, _) = s_basis(Representation::Spheric);
    for omega in SEPARATED {
        let error = PairBatchRequest::new(
            operator("int2c2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1]],
            cpu_options(Some(omega)),
        )
        .evaluate()
        .expect_err("a set range_omega must be refused by the pair batch surface");
        assert_names_range_omega(&error, "pair-batch");
    }
}

/// `QuartetBatchRequest` is `int2e_sph` — the Fock-build surface.
#[test]
fn quartet_batch_refuses_a_set_range_omega() {
    let (basis, _) = s_basis(Representation::Spheric);
    for omega in SEPARATED {
        let error = QuartetBatchRequest::new(
            operator("int2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1, 2, 3]],
            cpu_options(Some(omega)),
        )
        .evaluate()
        .expect_err("a set range_omega must be refused by the quartet batch surface");
        assert_names_range_omega(&error, "quartet-batch");
    }
}

/// The refusal must be about ω *specifically*, not a side effect of a tighter
/// gate: unset and explicitly-zero ω are the full Coulomb operator and must
/// still batch, producing byte-identical values.
///
/// `Some(0.0)` is not merely "accepted" — `g2e.c:4445` branches on
/// `omega == 0.`, so it has to be the same numbers as `None`, bit for bit.
#[test]
fn full_range_still_batches_on_every_surface() {
    let (basis, _) = s_basis(Representation::Spheric);

    let triple = |omega| {
        TripleBatchRequest::new(
            operator("int3c2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1, 2]],
            cpu_options(omega),
        )
        .evaluate()
        .expect("full-range triple batch")
        .values
    };
    assert_eq!(
        triple(None),
        triple(Some(0.0)),
        "range_omega = Some(0.0) is the full Coulomb operator and must batch identically"
    );

    let pair = |omega| {
        PairBatchRequest::new(
            operator("int2c2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1]],
            cpu_options(omega),
        )
        .evaluate()
        .expect("full-range pair batch")
        .values
    };
    assert_eq!(pair(None), pair(Some(0.0)));

    let quartet = |omega| {
        QuartetBatchRequest::new(
            operator("int2e_sph"),
            Representation::Spheric,
            &basis,
            [[0_u32, 1, 2, 3]],
            cpu_options(omega),
        )
        .evaluate()
        .expect("full-range quartet batch")
        .values
    };
    assert_eq!(quartet(None), quartet(Some(0.0)));
}

/// The `(s,s|s,s)` `int2e_cart` pilot: a set ω declines the fast path and the
/// batch completes on the scalar route.
///
/// Both halves matter. That it *succeeds* proves the pilot fell back rather
/// than refusing, and that its values equal the scalar `SessionRequest`'s
/// proves the fallback landed on the path that reads ω — a pilot that still ran
/// would return the full-range numbers and fail the second assert.
#[test]
fn the_ssss_pilot_declines_range_omega_and_falls_back_to_the_scalar_route() {
    let (basis, shells) = s_basis(Representation::Cart);
    let tuple = ShellTuple::try_from_iter([
        shells[0].clone(),
        shells[1].clone(),
        shells[2].clone(),
        shells[3].clone(),
    ])
    .unwrap();
    let int2e_cart = operator("int2e_cart");

    let scalar = |omega| {
        SessionRequest::new(
            int2e_cart,
            Representation::Cart,
            &basis,
            tuple.clone(),
            cpu_options(omega),
        )
        .query_workspace()
        .expect("scalar query_workspace")
        .evaluate()
        .expect("scalar evaluate")
        .tensor
        .owned_values
    };

    let batched = |omega| {
        let output = BatchRequest::new([SessionRequest::new(
            int2e_cart,
            Representation::Cart,
            &basis,
            tuple.clone(),
            cpu_options(omega),
        )])
        .evaluate_batch()
        .expect("a set range_omega must fall back, not fail, inside the ssss pilot");
        output.outputs[0].tensor.owned_values.clone()
    };

    for omega in SEPARATED {
        let separated = batched(Some(omega));
        assert_eq!(
            separated,
            scalar(Some(omega)),
            "the batched ssss route at range_omega={omega} must equal the scalar route"
        );
        // And it must not be the full-range answer, or the "equality" above
        // would be satisfied by two identically-wrong paths.
        assert_ne!(
            separated,
            scalar(None),
            "range_omega={omega} produced the full-range integral"
        );
    }

    // The pilot itself is untouched when ω is absent or zero: `None` and
    // `Some(0.0)` both take it, and are byte-identical to each other.
    let pilot = batched(None);
    assert_eq!(
        pilot,
        batched(Some(0.0)),
        "range_omega = Some(0.0) is full Coulomb and must take the same pilot as None"
    );
    // Against the scalar route the pilot agrees only to its own arithmetic —
    // it is a different kernel with a different summation order, so this is a
    // tolerance, not byte identity. That is also the tell that it really ran:
    // the ω cases above were byte-identical to the scalar route because they
    // *were* the scalar route.
    let scalar_full = scalar(None);
    let scale = scalar_full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let delta = pilot
        .iter()
        .zip(&scalar_full)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        delta <= 1e-13 * scale.max(1.0),
        "the full-range ssss pilot diverged from the scalar route: max |Δ| = {delta:.3e}"
    );
}
