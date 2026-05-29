//! Phase 22 (FND-01, D-03): gauge-origin env[1..3] slot raw<->plan round-trip.
//!
//! Verifies the Plan-22-01 PTR_COMMON_ORIG wiring without a consuming kernel:
//!   1. `eval_raw` on the non-zero fixture returns Ok (the live env[1..3] read does not error);
//!   2. the public env -> OperatorEnvParams.common_orig round-trip (mirroring the internal
//!      eval_raw read at raw.rs:604-615) yields Some(non-zero) and validates.
//!
//! Plain `#[test]` — NOT a vendor byte-identity parity test (D-03: slot verification only,
//! no kernel consumes common_orig until Phases 24/26).
#![cfg(feature = "cpu")]

use cintx_compat::raw::{PTR_COMMON_ORIG, RawApiId, eval_raw};
use cintx_core::cintxRsError; // re-exported from cintx_core (NOT cintx_compat); matches f12_oracle_parity.rs:394.
use cintx_oracle::fixtures::{
    COMMON_ORIG_FIXTURE_ORIGIN, build_h2o_sto3g, build_h2o_sto3g_common_orig,
};
use cintx_runtime::OperatorEnvParams; // crate-root re-export (lib.rs:19).
// Validator reached via the module path — the established convention (raw.rs:611/633); the
// validator family is NOT in the lib.rs `pub use validator::{...}` re-export. See <interfaces>.
use cintx_runtime::validator::validate_common_orig_env_params;

/// Read env[PTR_COMMON_ORIG..+3] into an OperatorEnvParams, mirroring the internal
/// eval_raw read at crates/cintx-compat/src/raw.rs:604-615 (the rinv_orig precedent).
/// This is the publicly observable round-trip of the raw env slot into the plan field.
fn common_orig_from_env(env: &[f64]) -> OperatorEnvParams {
    let mut params = OperatorEnvParams::default();
    if env.len() >= PTR_COMMON_ORIG + 3 {
        params.common_orig = Some([
            env[PTR_COMMON_ORIG],
            env[PTR_COMMON_ORIG + 1],
            env[PTR_COMMON_ORIG + 2],
        ]);
    }
    params
}

/// Single overlap shell-pair eval_raw call; returns the Result so the test can assert Ok.
/// Operator is a base 1e symbol (no consuming kernel reads common_orig yet, D-03).
fn run_ovlp_shell_pair(atm: &[i32], bas: &[i32], env: &[f64]) -> Result<(), cintxRsError> {
    let shls = [0_i32, 0_i32]; // shell 0 x shell 0 (O 1s, l=0 -> 1x1 output)
    let mut out = vec![0.0_f64; 1];
    // SAFETY: atm/bas/env are well-formed by construction in the fixture builders;
    // shls=[0,0] are valid shell indices. Mirrors one_electron_parity.rs:243-246.
    unsafe {
        eval_raw(
            RawApiId::INT1E_OVLP_SPH,
            Some(&mut out),
            None,
            &shls,
            atm,
            bas,
            env,
            None,
            None,
        )?;
    }
    Ok(())
}

#[test]
fn common_orig_nonzero_fixture_roundtrips_into_plan() {
    let (atm, bas, env) = build_h2o_sto3g_common_orig();

    // (1) Live path: eval_raw on the non-zero fixture must not error.
    run_ovlp_shell_pair(&atm, &bas, &env)
        .expect("eval_raw must succeed on the non-zero gauge-origin fixture");

    // (2) Public env -> plan slot round-trip (mirrors eval_raw raw.rs:604-615).
    let params = common_orig_from_env(&env);
    assert_eq!(
        params.common_orig,
        Some(COMMON_ORIG_FIXTURE_ORIGIN),
        "non-zero env[1..3] must round-trip into operator_env_params.common_orig"
    );
    // Prove it is NOT the default origin (the point of a non-zero fixture, CONTEXT line 103):
    assert_ne!(params.common_orig, Some([0.0, 0.0, 0.0]));

    // Validator accepts the populated, finite origin.
    validate_common_orig_env_params("int1e_ovlp_sph", &params)
        .expect("non-zero finite common_orig must validate");
}

#[test]
fn common_orig_base_fixture_reads_zero_default_not_skipped() {
    // Base fixture leaves env[1..3] at the all-zero default; the read is unconditional (D-02).
    let (atm, bas, env) = build_h2o_sto3g();

    run_ovlp_shell_pair(&atm, &bas, &env).expect("eval_raw must succeed on the base fixture");

    let params = common_orig_from_env(&env);
    assert_eq!(
        params.common_orig,
        Some([0.0, 0.0, 0.0]),
        "the slot is READ (not skipped) — a zero origin reads back as Some([0,0,0])"
    );
    validate_common_orig_env_params("int1e_ovlp_sph", &params)
        .expect("zero finite common_orig must validate");
}
