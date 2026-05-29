//! grids family ROCm validation + pre-existing blocker lock (quick task 260529-twi).
//!
//! ⚠ The CubeCL `grids_scalar_kernel` GPU device port (scalar nuclear-like
//! `int1e_grids_sph`, ported to a `#[cube(launch)]` device kernel + dispatched on
//! the ROCm `HipRuntime` in quick-260529-twi) CANNOT be validated through this
//! oracle crate's usual `eval_raw`-vs-vendor pattern, because of a PRE-EXISTING
//! UPSTREAM BLOCKER unrelated to the GPU port:
//!
//!   `eval_raw(RawApiId::Symbol("int1e_grids_sph"), …)` rejects the grids
//!   4-element shell tuple with `InvalidShellTuple { expected: 2, got: 4 }`.
//!
//! This same failure makes ALL `unstable_source_parity.rs::grids_parity`
//! CPU-vs-vendor tests fail at baseline on this branch — it is baseline noise,
//! NOT a regression introduced by the device port. Fixing the eval_raw /
//! ExecutionPlan 4-shell-tuple handling is a separate, out-of-scope task (the
//! user explicitly chose to document the blocker rather than fix the wiring).
//!
//! REAL ROCm validation of the grids device kernel therefore lives as a DIRECT
//! device-vs-host parity test INSIDE the cintx-cubecl crate (where the kernel +
//! host reference are accessible without eval_raw):
//!
//!   crates/cintx-cubecl/src/kernels/unstable/grids.rs
//!     → tests::test_grids_device_vs_host_rocm
//!   Run: CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!     cargo test -p cintx-cubecl --features rocm,unstable-source-api \
//!     --lib kernels::unstable::grids::tests::test_grids_device_vs_host_rocm -- --ignored
//!   Result (AMD gfx1152): cases=64 mismatch_count=0 any_nonzero=true.
//!
//! The test below LOCKS the blocker: it asserts the eval_raw rejection still
//! reproduces. When the upstream 4-shell-tuple wiring is eventually fixed, this
//! test will fail loudly — the signal to wire a real grids ROCm vendor oracle here.

#![cfg(feature = "rocm")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NGRIDS, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_GRIDS, RawApiId, eval_raw,
};

/// Compact O/O s-s grids fixture (2 shells, 2 grid points) — enough to drive the
/// grids raw-API entry point to the shell-tuple validation.
fn build_compact_grids() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];
    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_exp);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_coeff);

    let ngrids = 2usize;
    let ptr_grids_val = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, 0.0]);
    env.extend_from_slice(&[0.5, -0.5, 0.5]);
    env[NGRIDS] = ngrids as f64;
    env[PTR_GRIDS] = ptr_grids_val as f64;

    let mut atm = vec![0_i32; ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for s in 0..2 {
        bas[s * BAS_SLOTS + ATOM_OF] = 0;
        bas[s * BAS_SLOTS + ANG_OF] = 0;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }

    (atm, bas, env)
}

/// Locks the pre-existing grids `eval_raw` blocker. `#[ignore]` so it only runs
/// under the explicit ROCm-oracle invocation alongside the in-crate device test.
#[test]
#[ignore]
fn test_grids_eval_raw_blocked_pre_existing() {
    let (atm, bas, env) = build_compact_grids();
    // grids uses a 4-element shell tuple — the source of the upstream rejection.
    let shls: [i32; 4] = [0, 1, 0, 1];
    let mut out = vec![0.0_f64; 64];
    let res = unsafe {
        eval_raw(
            RawApiId::Symbol("int1e_grids_sph"),
            Some(&mut out),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    };
    let err = res.expect_err(
        "PRE-EXISTING grids blocker appears FIXED: eval_raw int1e_grids_sph now succeeds. \
         Wire a real grids ROCm vendor oracle here and remove this lock.",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("InvalidShellTuple") || msg.contains("expected: 2"),
        "grids eval_raw failed with an UNEXPECTED error (blocker may have changed shape): {msg}"
    );
    eprintln!(
        "grids_random_rocm_parity: pre-existing eval_raw blocker confirmed ({msg}). \
         Real grids ROCm device-vs-host validation: cintx-cubecl \
         kernels::unstable::grids::tests::test_grids_device_vs_host_rocm."
    );
}
