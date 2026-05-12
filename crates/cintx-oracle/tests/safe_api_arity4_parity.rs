//! Safe-API arity-4 parity tests for the 4 ROADMAP-named arity-4 operators.
//!
//! Each per-symbol `#[test]` function iterates the full Cartesian product of the
//! H2O/STO-3G 5 shells (625 quartets) and asserts byte-identity vs vendored
//! libcint 6.1.3 at the Phase 15 unified tolerance (atol=1e-12, rtol=0.0).
//!
//! `int4c1e_*` tests are individually `#[cfg(feature = "with-4c1e")]`-gated so this
//! file compiles under every profile; the `int2e_*` tests run under the base profile.
//!
//! Arity-4 cintx kernels write F-order matching vendor output directly — no
//! transpose needed (precedent: `crates/cintx-oracle/src/compare.rs:787-797`).
//!
//! See `.planning/phases/18-sessionrequest-arity-ge3-dispatch/18-CONTEXT.md` for
//! decision rationale (D-06, D-12, D-13, D-14, D-15).

// Module gate widened to allow `--features rocm` (without cpu) per Phase 16-04 pattern.
// PATTERNS.md §arity4 anti-pattern: do NOT add a module-level with-4c1e cfg here —
// it would break the int2e_* tests under the base profile. The with-4c1e gating is
// applied per-test on the int4c1e_* functions only (see those tests below).
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance constants (Phase 15 unified: atol=1e-12, rtol=0.0 — CONTEXT.md D-15)
// ─────────────────────────────────────────────────────────────────────────────

const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;
const N_SHELLS: usize = 5;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G raw fixture (PTR_ENV_START-aware version, copied verbatim from
// safe_api_arity2_parity.rs:36-160).
//
// CRITICAL: libcint reserves env[0..PTR_ENV_START] for global parameters.
// User data MUST start at PTR_ENV_START=20.
// ─────────────────────────────────────────────────────────────────────────────

fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];

    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START]; // zeros for reserved slots

    let o_coord_ptr = env.len() as i32; // 20
    env.extend_from_slice(&o_coord);

    let h1_coord_ptr = env.len() as i32; // 23
    env.extend_from_slice(&h1_coord);

    let h2_coord_ptr = env.len() as i32; // 26
    env.extend_from_slice(&h2_coord);

    let zeta_ptr = env.len() as i32; // 29
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32; // 30
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; // 33
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32; // 36
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; // 39
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32; // 42
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; // 45
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32; // 48
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; // 51
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];

    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    // Shell 0: O 1s (l=0, 3 prim, 1 ctr)
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    // Shell 1: O 2s (l=0, 3 prim, 1 ctr)
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    // Shell 2: O 2p (l=1, 3 prim, 1 ctr)
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    // Shell 3: H1 1s (l=0, 3 prim, 1 ctr)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    // Shell 4: H2 1s (l=0, 3 prim, 1 ctr)
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G safe-API BasisSet construction helper.
//
// Numeric values match build_h2o_sto3g() exactly so vendor comparisons are valid.
// Returns (BasisSet, Vec<Arc<Shell>>) — the caller builds ShellTuple per-call.
// ─────────────────────────────────────────────────────────────────────────────

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o = Atom::try_new(8, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atom_h2 = Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G exponents and coefficients (Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657, 1969).
    // These values match build_h2o_sto3g() exactly so vendor comparisons are valid.

    // O 1s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
    let shell_o1s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    // O 2s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
    let shell_o2s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[-0.09996723, 0.39951283, 0.70011547]),
    ).unwrap());

    // O 2p: atom_idx=0, l=1, nprim=3, nctr=1, kappa=0
    let shell_o2p = Arc::new(Shell::try_new(
        0, 1, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[0.15591627, 0.60768372, 0.39195739]),
    ).unwrap());

    // H1 1s: atom_idx=1, l=0, nprim=3, nctr=1, kappa=0
    let shell_h1_1s = Arc::new(Shell::try_new(
        1, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    // H2 1s: atom_idx=2, l=0, nprim=3, nctr=1, kappa=0
    let shell_h2_1s = Arc::new(Shell::try_new(
        2, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    let shells = vec![shell_o1s, shell_o2s, shell_o2p, shell_h1_1s, shell_h2_1s];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}

// ─────────────────────────────────────────────────────────────────────────────
// Arity-agnostic per-tuple safe-API buffer collector.
//
// Drives `SessionRequest::evaluate` — the safe-API path that Plan 17-02 wired
// to the real CubeClExecutor. Returns the raw `owned_values` buffer for direct
// byte-to-byte comparison against vendor output. NO transpose required for
// arity ≥ 3 (precedent: compare.rs:787-797 for int2e_sph).
//
// `tuple_shells` must have length within SHELL_TUPLE_CAPACITY=4 (RESEARCH.md).
// For arity-4 callers, `tuple_shells.len() == 4`.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_safe_api_tuple_buffer(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    tuple_shells: &[Arc<Shell>],
) -> Vec<f64> {
    let shell_tuple = ShellTuple::try_from_iter(tuple_shells.iter().cloned())
        .expect("tuple within SHELL_TUPLE_CAPACITY=4");
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
        .expect("evaluate must succeed for arity-4 dispatch");
    output.tensor.owned_values
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance comparison helper (Phase 15 unified: atol=1e-12, rtol=0.0).
//
// Copied verbatim from safe_api_arity2_parity.rs:300-321 / one_electron_parity.rs.
// ─────────────────────────────────────────────────────────────────────────────

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "output length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-symbol arity-4 parity tests
//
// Each test iterates the full 5⁴ = 625 quartet sweep over the H2O/STO-3G basis
// and asserts byte-identity vs vendored libcint at atol=1e-12, rtol=0.0. The
// `any_nonzero` sentinel guards against zero-fill regressions (PATTERNS.md
// §Shared Patterns).
//
// Per CONTEXT.md D-14: per-symbol `#[test]` functions (NOT a parametric loop)
// so CI failure messages are directly bisectable to the offending operator.
//
// Per CONTEXT.md D-15: atol=1e-12, rtol=0.0 — Phase 15 unified tolerance.
//
// Per PATTERNS.md §Shared Patterns: NO transpose — arity-4 cintx and vendor
// kernels write F-order, agreeing byte-for-byte without conversion
// (precedent: compare.rs:787-797 for int2e_sph).
//
// OperatorId mapping (post-Plan-18-01 manifest expansion):
//   int2e_cart            OperatorId::new(9)    // unchanged
//   int2e_sph             OperatorId::new(10)   // unchanged
//   int4c1e_cart          OperatorId::new(24)   // +2 from Plan 18-01 R1 shift
//   int4c1e_sph           OperatorId::new(25)   // +2 from Plan 18-01 R1 shift
//
// Vendor wrappers (all pre-existing; no new wrappers needed for arity-4):
//   int2e_cart            vendor_int2e_cart    (crates/cintx-oracle/src/vendor_ffi.rs)
//   int2e_sph             vendor_int2e_sph     (crates/cintx-oracle/src/vendor_ffi.rs)
//   int4c1e_cart          vendor_int4c1e_cart  (crates/cintx-oracle/src/vendor_ffi.rs)
//   int4c1e_sph           vendor_int4c1e_sph   (crates/cintx-oracle/src/vendor_ffi.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_int2e_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shells[i].ao_per_shell();
                    let nj = shells[j].ao_per_shell();
                    let nk = shells[k].ao_per_shell();
                    let nl = shells[l].ao_per_shell();
                    let n_elem = ni * nj * nk * nl;

                    let safe_out = collect_safe_api_tuple_buffer(
                        OperatorId::new(9),
                        Representation::Cart,
                        &basis,
                        &[
                            shells[i].clone(),
                            shells[j].clone(),
                            shells[k].clone(),
                            shells[l].clone(),
                        ],
                    );

                    let mut vendor_out = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    cintx_oracle::vendor_ffi::vendor_int2e_cart(
                        &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                    );

                    if safe_out.iter().any(|&v| v.abs() > 1e-18)
                        || vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    {
                        any_nonzero = true;
                    }
                    total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                    tuples_checked += 1;
                }
            }
        }
    }

    assert!(
        any_nonzero,
        "int2e_cart safe-API outputs are all zeros over {tuples_checked} quartets"
    );
    assert_eq!(
        total_mismatches, 0,
        "int2e_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} quartets"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int2e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shells[i].ao_per_shell();
                    let nj = shells[j].ao_per_shell();
                    let nk = shells[k].ao_per_shell();
                    let nl = shells[l].ao_per_shell();
                    let n_elem = ni * nj * nk * nl;

                    let safe_out = collect_safe_api_tuple_buffer(
                        OperatorId::new(10),
                        Representation::Spheric,
                        &basis,
                        &[
                            shells[i].clone(),
                            shells[j].clone(),
                            shells[k].clone(),
                            shells[l].clone(),
                        ],
                    );

                    let mut vendor_out = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    cintx_oracle::vendor_ffi::vendor_int2e_sph(
                        &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                    );

                    if safe_out.iter().any(|&v| v.abs() > 1e-18)
                        || vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    {
                        any_nonzero = true;
                    }
                    total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                    tuples_checked += 1;
                }
            }
        }
    }

    assert!(
        any_nonzero,
        "int2e_sph safe-API outputs are all zeros over {tuples_checked} quartets"
    );
    assert_eq!(
        total_mismatches, 0,
        "int2e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} quartets"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// int4c1e_* per-symbol parity tests (CONTEXT.md D-06 + D-12)
//
// Each test stacks per-test attributes verbatim per PATTERNS.md §arity4:
//   #[test]
//   #[cfg(feature = "with-4c1e")]   <-- per-test, NOT module-level
//   #[cfg(has_vendor_libcint)]
//
// Both `#[cfg(...)]` attributes stack additively — the test compiles only when
// both `feature = "with-4c1e"` AND `has_vendor_libcint` are active. PATTERNS.md
// §Pitfall 4 forbids a module-level with-4c1e gate: it would break the int2e_*
// tests under the base profile.
//
// Precedent for stacked cfg attributes: crates/cintx-oracle/tests/oracle_gate_closure.rs:737-739.
//
// Phase 11 D-09 keeps the complex int4c1e variant out of scope — only the
// cart/sph wrappers are exercised here.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "with-4c1e")]
#[cfg(has_vendor_libcint)]
fn test_int4c1e_cart_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Cart);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shells[i].ao_per_shell();
                    let nj = shells[j].ao_per_shell();
                    let nk = shells[k].ao_per_shell();
                    let nl = shells[l].ao_per_shell();
                    let n_elem = ni * nj * nk * nl;

                    let safe_out = collect_safe_api_tuple_buffer(
                        OperatorId::new(24),
                        Representation::Cart,
                        &basis,
                        &[
                            shells[i].clone(),
                            shells[j].clone(),
                            shells[k].clone(),
                            shells[l].clone(),
                        ],
                    );

                    let mut vendor_out = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    cintx_oracle::vendor_ffi::vendor_int4c1e_cart(
                        &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                    );

                    if safe_out.iter().any(|&v| v.abs() > 1e-18)
                        || vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    {
                        any_nonzero = true;
                    }
                    total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                    tuples_checked += 1;
                }
            }
        }
    }

    assert!(
        any_nonzero,
        "int4c1e_cart safe-API outputs are all zeros over {tuples_checked} quartets"
    );
    assert_eq!(
        total_mismatches, 0,
        "int4c1e_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} quartets"
    );
}

#[test]
#[cfg(feature = "with-4c1e")]
#[cfg(has_vendor_libcint)]
fn test_int4c1e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shells[i].ao_per_shell();
                    let nj = shells[j].ao_per_shell();
                    let nk = shells[k].ao_per_shell();
                    let nl = shells[l].ao_per_shell();
                    let n_elem = ni * nj * nk * nl;

                    let safe_out = collect_safe_api_tuple_buffer(
                        OperatorId::new(25),
                        Representation::Spheric,
                        &basis,
                        &[
                            shells[i].clone(),
                            shells[j].clone(),
                            shells[k].clone(),
                            shells[l].clone(),
                        ],
                    );

                    let mut vendor_out = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    cintx_oracle::vendor_ffi::vendor_int4c1e_sph(
                        &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env,
                    );

                    if safe_out.iter().any(|&v| v.abs() > 1e-18)
                        || vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    {
                        any_nonzero = true;
                    }
                    total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                    tuples_checked += 1;
                }
            }
        }
    }

    assert!(
        any_nonzero,
        "int4c1e_sph safe-API outputs are all zeros over {tuples_checked} quartets"
    );
    assert_eq!(
        total_mismatches, 0,
        "int4c1e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} quartets"
    );
}
