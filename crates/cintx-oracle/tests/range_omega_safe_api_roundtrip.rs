//! D-PBC-24 — the SAFE API honours `range_omega`, and agrees with the raw path.
//!
//! `.with_range_omega(ω)` → `ExecutionOptions::range_omega` → `query_workspace`
//! (which sizes the short-range doubled Rys roots) → `operator_env_params` →
//! the `CINTg0_2e` prologue. That chain has to produce exactly what a raw
//! caller writing `env[PTR_RANGE_OMEGA]` gets, or the two API surfaces have
//! quietly become two methods.
//!
//! Also pins the two contracts a caller can trip:
//!
//! * ω is part of the workspace query, so changing it afterwards is backend
//!   contract drift and is refused, not silently honoured;
//! * ω on an operator cintx has not implemented range separation for is a typed
//!   refusal, not a full-range evaluation.
//!
//! Plain `#[test]`s — no vendor build needed. The numeric gate against libcint
//! is `range_omega_parity.rs`.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RANGE_OMEGA, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_rs::builder::SessionBuilder;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

/// Two p-shells on two centres: `int2c2e` `rys_order = (1+1)/2 + 1 = 2`, so
/// short range doubles the roots to 4 and the workspace has to grow.
const L: i32 = 1;
const EXPS: [f64; 2] = [0.9, 0.6];
const COORDS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.3, 0.0, 1.7]];

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn safe_basis() -> (BasisSet, Vec<Arc<Shell>>) {
    let atoms: Vec<Atom> = COORDS
        .iter()
        .map(|&c| Atom::try_new(2, c, NuclearModel::Point, None, None).unwrap())
        .collect();
    let atoms = Arc::from(atoms.into_boxed_slice());

    let shells: Vec<Arc<Shell>> = (0..2)
        .map(|s| {
            Arc::new(
                Shell::try_new(
                    s as u32,
                    L as u8,
                    1,
                    1,
                    0,
                    Representation::Spheric,
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

/// The same fixture in `atm`/`bas`/`env` form, for the raw comparison.
fn raw_fixture(omega: Option<f64>) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0_f64; PTR_ENV_START];
    if let Some(omega) = omega {
        env[PTR_RANGE_OMEGA] = omega;
    }

    let mut coord_ptr = [0_i32; 2];
    for (s, ptr) in coord_ptr.iter_mut().enumerate() {
        *ptr = env.len() as i32;
        env.extend_from_slice(&COORDS[s]);
    }
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut exp_ptr = [0_i32; 2];
    let mut coeff_ptr = [0_i32; 2];
    for s in 0..2 {
        exp_ptr[s] = env.len() as i32;
        env.push(EXPS[s]);
        coeff_ptr[s] = env.len() as i32;
        env.push(1.0);
    }

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for s in 0..2 {
        atm[s * ATM_SLOTS + CHARGE_OF] = 2;
        atm[s * ATM_SLOTS + PTR_COORD] = coord_ptr[s];
        atm[s * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[s * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for s in 0..2 {
        bas[s * BAS_SLOTS + ATOM_OF] = s as i32;
        bas[s * BAS_SLOTS + ANG_OF] = L;
        bas[s * BAS_SLOTS + NPRIM_OF] = 1;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr[s];
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr[s];
    }

    (atm, bas, env)
}

fn int2c2e_sph_id() -> cintx_core::OperatorId {
    Resolver::descriptor_by_symbol("int2c2e_sph")
        .expect("int2c2e_sph is a registered operator")
        .id
}

fn eval_safe(omega: Option<f64>) -> Vec<f64> {
    let (basis, shells) = safe_basis();
    let tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()]).unwrap();
    let mut builder = SessionBuilder::new(
        int2c2e_sph_id(),
        Representation::Spheric,
        &basis,
        tuple.clone(),
    );
    if let Some(omega) = omega {
        builder = builder.with_range_omega(omega);
    }
    let output = builder
        .build()
        .query_workspace()
        .expect("query_workspace")
        .evaluate()
        .expect("evaluate");
    output.tensor.owned_values
}

fn eval_raw_path(omega: Option<f64>) -> Vec<f64> {
    let (atm, bas, env) = raw_fixture(omega);
    let n = 9; // (2l+1)^2 for l = 1
    let mut out = vec![0.0_f64; n];
    unsafe {
        eval_raw(
            RawApiId::INT2C2E_SPH,
            Some(&mut out),
            None,
            &[0_i32, 1],
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .expect("eval_raw int2c2e_sph");
    }
    out
}

/// The two API surfaces must produce the SAME numbers under the same ω.
///
/// Byte identity, not a tolerance: both go through the identical planner,
/// prologue and kernel — only the way ω arrives differs (`ExecutionOptions`
/// versus `env[8]`). Any divergence here is a plumbing bug, not arithmetic.
#[test]
fn the_safe_api_and_the_raw_env_slot_agree_on_every_omega() {
    for omega in [None, Some(0.3), Some(0.8), Some(-0.3), Some(-0.8)] {
        let safe = eval_safe(omega);
        let raw = eval_raw_path(omega);
        assert_eq!(
            safe, raw,
            "safe API and raw env[8] disagree at range_omega={omega:?}"
        );
    }
}

/// Range separation must actually change the numbers — otherwise the plumbing
/// could be inert and every test above would still pass.
#[test]
fn range_separation_changes_the_result() {
    let full = eval_safe(None);
    for omega in [0.3_f64, 0.8, -0.3, -0.8] {
        let separated = eval_safe(Some(omega));
        let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let delta = full
            .iter()
            .zip(&separated)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            delta > 1e-3 * scale,
            "range_omega={omega} produced (almost) the full-range result: \
             max |Δ| = {delta:.3e} against scale {scale:.3e}"
        );
    }
    // And an explicit zero is the full-range operator, exactly (g2e.c:4445
    // branches on `omega == 0.`).
    assert_eq!(eval_safe(Some(0.0)), full);
}

/// Short range doubles `nrys_roots` for `rys_order <= 3`, which SIZES the
/// workspace — so ω belongs to the query, and the safe API's `WorkspacePlan`
/// has to show it. (Changing ω between query and evaluate is the D-08 contract
/// drift `planning_matches` rejects; that half is pinned by
/// `cintx_runtime::planner`'s
/// `short_range_doubles_the_queried_rys_roots_and_query_evaluate_agree`.)
#[test]
fn short_range_asks_for_a_bigger_workspace_through_the_safe_api() {
    let (basis, shells) = safe_basis();
    let tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()]).unwrap();

    let queried = SessionRequest::new(
        int2c2e_sph_id(),
        Representation::Spheric,
        &basis,
        tuple.clone(),
        ExecutionOptions {
            range_omega: Some(-0.8),
            ..ExecutionOptions::default()
        },
    );
    let sr_query = queried.query_workspace().expect("SR query_workspace");
    assert_eq!(
        sr_query.workspace().rys_roots,
        4,
        "int2c2e over two p-shells is rys_order 2; short range must query 4 roots"
    );

    let full_query = SessionRequest::new(
        int2c2e_sph_id(),
        Representation::Spheric,
        &basis,
        tuple,
        ExecutionOptions::default(),
    )
    .query_workspace()
    .expect("full-range query_workspace");
    assert_eq!(full_query.workspace().rys_roots, 2);
    assert!(
        sr_query.workspace().required_bytes > full_query.workspace().required_bytes,
        "the doubled roots must show up in the workspace request"
    );
}

/// D-PBC-24 P2-3 — `range_omega` under `PrecisionKind::F32`.
///
/// `rys_roots_range_separated` is `f64`-only, and all three prologues call it in
/// `f64` regardless of `plan.precision`: the host arm accumulates `cart_blocks:
/// Vec<f64>` and casts once at the staging write. That is exactly what the
/// full-range host arms already do, so f32 + range separation is believed
/// correct by construction — but until this test it was believed, not measured,
/// and "the f64 root solver silently produced an f32-truncated ω" is not a
/// failure the f64 gates above could ever see.
///
/// Reference is the f64 result on the SAME ω, not the full-range one: the
/// question here is whether f32 costs precision, not whether ω arrived (the
/// tests above settle that). Tolerance is the family's published f32 floor.
#[test]
fn range_separation_survives_the_f32_precision_path() {
    use cintx_oracle::compare::f32_tolerance_for_family;

    let (basis, shells) = safe_basis();
    let tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()]).unwrap();
    let tol = f32_tolerance_for_family("2c2e");

    for omega in [
        None,
        Some(0.0),
        Some(0.3),
        Some(0.8),
        Some(-0.3),
        Some(-0.8),
    ] {
        let mut builder = SessionBuilder::new(
            int2c2e_sph_id(),
            Representation::Spheric,
            &basis,
            tuple.clone(),
        );
        if let Some(omega) = omega {
            builder = builder.with_range_omega(omega);
        }
        let got = builder
            .build()
            .query_workspace()
            .expect("f32 query_workspace under a set range_omega")
            .evaluate_generic::<f32>()
            .expect("f32 evaluate under a set range_omega")
            .tensor
            .owned_values;

        let want = eval_safe(omega);
        assert_eq!(
            got.len(),
            want.len(),
            "the f32 path must produce the same block shape at range_omega={omega:?}"
        );

        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0_f64;
        for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let diff = (g as f64 - w).abs();
            worst = worst.max(diff);
            assert!(
                diff <= tol.atol + tol.rtol * w.abs().max(scale * tol.zero_threshold),
                "f32 range_omega={omega:?} idx={idx}: f32={:.9e} f64={w:.17e} diff={diff:.3e} \
                 (atol={:.0e} rtol={:.0e})",
                g as f64,
                tol.atol,
                tol.rtol
            );
        }

        // The f32 block must not be a truncated FULL-RANGE block: an ω that
        // reached the f64 planner but not the f32 staging write would land here
        // inside the tolerance above and nowhere else.
        if let Some(omega) = omega
            && omega != 0.0
        {
            let full = eval_safe(None);
            let moved = got
                .iter()
                .zip(&full)
                .fold(0.0_f64, |m, (&g, &f)| m.max((g as f64 - f).abs()));
            assert!(
                moved > 1e-3 * scale,
                "f32 range_omega={omega} produced the full-range block: \
                 max |Δ| = {moved:.3e} against scale {scale:.3e}"
            );
        }
    }
}

/// A set, non-zero ω on an operator with no range-separated kernel is a typed
/// refusal. It must NEVER fall through to the full-range one: that runs,
/// converges, and is silently a different method.
#[test]
fn omega_on_an_unsupported_operator_is_refused_not_ignored() {
    let (basis, shells) = safe_basis();
    let tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()]).unwrap();
    let ovlp = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
        .expect("int1e_ovlp_sph is registered")
        .id;

    let err = SessionBuilder::new(ovlp, Representation::Spheric, &basis, tuple)
        .with_range_omega(-0.8)
        .build()
        .query_workspace()
        .expect_err("a set omega on int1e_ovlp must be refused");
    let text = format!("{err}");
    assert!(
        text.contains("range_omega") || text.contains("int2e"),
        "the refusal must say what it is refusing and what IS supported, got: {text}"
    );
}

/// A non-finite ω is rejected before it can reach `theta = ω²/(ω² + a0)` and
/// poison every Rys weight with a `NaN`.
#[test]
fn a_non_finite_omega_is_rejected() {
    let (basis, shells) = safe_basis();
    let tuple = ShellTuple::try_from_iter([shells[0].clone(), shells[1].clone()]).unwrap();

    for omega in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = SessionBuilder::new(
            int2c2e_sph_id(),
            Representation::Spheric,
            &basis,
            tuple.clone(),
        )
        .with_range_omega(omega)
        .build()
        .query_workspace()
        .expect_err("a non-finite range_omega must be rejected");
        let text = format!("{err}");
        assert!(
            text.contains("PTR_RANGE_OMEGA"),
            "the refusal must name the env slot, got: {text}"
        );
    }
}
