//! W5-00 — the manifest's `unsupported_policy` must match reality.
//!
//! 45 manifest rows carry `stability = "stable"`, a declared `forms` entry, and
//! `oracle_covered = false` while their kernel returns `UnsupportedApi`. Before
//! W5-00 the audit could not tell those apart from rows that are implemented but
//! merely unproven, so `manifest-audit --check-lock` was red with no actionable
//! signal.
//!
//! This test is the structural guard on both directions of that claim:
//!
//!   * every row marked `fail_closed` MUST actually fail closed — otherwise the
//!     manifest is hiding a working family behind a rejection, or (worse) hiding
//!     a family that quietly started returning numbers nobody has proven;
//!   * no row marked `fail_closed` may also be `oracle_covered = true` — that
//!     contradiction is checked in `xtask manifest-audit`, and asserted here too
//!     so it fails at test time rather than only at gate time.
//!
//! It is the reason the parent plan's W0-06 fail-open class of defect (a `_ => {}`
//! fall-through silently serving the wrong family) cannot be reintroduced without
//! a test going red.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA,
    RawApiId, eval_raw,
};
use cintx_ops::resolver::Resolver;

const ANG: i32 = 2;
const NCTR: usize = 2;

/// A `d`-shell `nctr = 2` fixture with non-zero gauge and rinv origins, so the
/// origin-reading families cannot pass by accident.
fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    env[PTR_COMMON_ORIG..PTR_COMMON_ORIG + 3].copy_from_slice(&[0.23, -0.41, 0.17]);
    env[PTR_RINV_ORIG..PTR_RINV_ORIG + 3].copy_from_slice(&[-0.11, 0.29, 0.05]);
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.35, 0.8]);

    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0; 4 * BAS_SLOTS];
    for shell in 0..4 {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = (shell % 2) as i32;
        bas[offset + ANG_OF] = ANG;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = NCTR as i32;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

/// `int1e_ecp_iprinv_spinor` reaches its R5 rejection at `ecp.rs:2047` only after
/// the ECP env validation, which this non-ECP fixture cannot satisfy. It still
/// fails closed — just with `InvalidEnvParameter` first — so it is asserted as
/// "errors", not as "errors with UnsupportedApi". Its real gate is W5-04.
const ECP_FIXTURE_EXEMPT: &[&str] = &["int1e_ecp_iprinv_spinor"];

fn probe(symbol: &'static str, arity: usize, rank: usize) -> Result<(), String> {
    let (atm, bas, env) = fixture();
    let d = NCTR * (4 * ANG as usize + 2);
    // Generous: the call must be rejected before any write, so an oversized
    // buffer cannot mask a rejection as a BufferTooSmall.
    let len = rank.max(1) * d.pow(arity as u32) * 2;
    let mut out = vec![0.0; len];
    let shls: Vec<i32> = (0..arity as i32).collect();
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn every_fail_closed_row_actually_fails_closed() {
    let mut leaked = Vec::new();
    let mut contradictions = Vec::new();

    for entry in Resolver::manifest() {
        let Some(policy) = entry.unsupported_policy.as_ref() else {
            continue;
        };
        // `no_upstream_oracle` rows DO evaluate — what they lack is a vendored
        // reference to prove against (an unconditional libcint stub). They are
        // asserted by `no_upstream_oracle_rows_still_evaluate` instead.
        if policy.policy == "no_upstream_oracle" {
            continue;
        }
        if entry.oracle_covered {
            contradictions.push(format!(
                "{}: oracle_covered=true AND unsupported_policy={} (owner {})",
                entry.symbol_name, policy.policy, policy.owner
            ));
        }

        let rank: usize = entry.component_rank.parse().unwrap_or(1);
        match probe(entry.symbol_name, entry.arity as usize, rank) {
            Err(msg) => {
                if !ECP_FIXTURE_EXEMPT.contains(&entry.symbol_name)
                    && !msg.contains("unsupported api")
                {
                    leaked.push(format!(
                        "{}: expected UnsupportedApi, got `{msg}`",
                        entry.symbol_name
                    ));
                }
            }
            Ok(()) => leaked.push(format!(
                "{}: marked fail_closed (owner {}, {}) but RETURNED DATA — either \
                 implement+prove it and drop the policy, or find what regressed",
                entry.symbol_name, policy.owner, policy.reason
            )),
        }
    }

    assert!(
        contradictions.is_empty(),
        "manifest rows claim coverage AND a fail-closed policy:\n  {}",
        contradictions.join("\n  ")
    );
    assert!(
        leaked.is_empty(),
        "{} fail-closed row(s) do not fail closed:\n  {}",
        leaked.len(),
        leaked.join("\n  ")
    );
}

/// The converse: a row with no policy and `oracle_covered = true` must not be
/// secretly rejecting. Restricted to the spinor derivative rows Wave 5 touches,
/// because a blanket sweep would need a valid fixture for every family in the
/// manifest (ECP basis, grids, F12 zeta, …) and would fail for fixture reasons
/// rather than coverage reasons.
#[test]
fn covered_rows_wave5_touches_are_not_secretly_rejecting() {
    const WATCHED: &[&str] = &[
        "int3c2e_ip1_spinor",
        "int3c2e_ip2_spinor",
        "int2c2e_ip1_spinor",
        "int2c2e_ip2_spinor",
        "int3c1e_spinor",
        "int3c1e_ip1_spinor",
        "int3c1e_iprinv_spinor",
    ];
    let mut rejecting = Vec::new();
    for symbol in WATCHED {
        let Ok(desc) = Resolver::descriptor_by_symbol(symbol) else {
            continue;
        };
        let entry = desc.entry;
        if !entry.oracle_covered {
            continue;
        }
        let rank: usize = entry.component_rank.parse().unwrap_or(1);
        if let Err(msg) = probe(entry.symbol_name, entry.arity as usize, rank) {
            rejecting.push(format!("{symbol}: oracle_covered=true but errors — {msg}"));
        }
    }
    assert!(
        rejecting.is_empty(),
        "rows claim coverage but reject on the d-shell nctr=2 fixture:\n  {}",
        rejecting.join("\n  ")
    );
}

/// The `no_upstream_oracle` rows must keep WORKING even though they can never be
/// oracle-gated. If one starts erroring, the manifest is now lying in the other
/// direction — it would be claiming an implementation that no longer exists, with
/// no vendor test able to notice.
#[test]
fn no_upstream_oracle_rows_still_evaluate() {
    let mut broken = Vec::new();
    for entry in Resolver::manifest() {
        let Some(policy) = entry.unsupported_policy.as_ref() else {
            continue;
        };
        if policy.policy != "no_upstream_oracle" {
            continue;
        }
        // int2c2e_ip1ip2_spinor is BOTH upstream-stubbed and rejected by cintx
        // (center_2c2e.rs:829); implementing it could never be proven, so it is
        // deliberately left unimplemented.
        if entry.symbol_name == "int2c2e_ip1ip2_spinor" {
            continue;
        }
        let rank: usize = entry.component_rank.parse().unwrap_or(1);
        if let Err(msg) = probe(entry.symbol_name, entry.arity as usize, rank) {
            broken.push(format!("{}: {msg}", entry.symbol_name));
        }
    }
    assert!(
        broken.is_empty(),
        "no_upstream_oracle rows must still evaluate:\n  {}",
        broken.join("\n  ")
    );
}
