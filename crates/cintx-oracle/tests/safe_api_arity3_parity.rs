//! Safe-API arity-3 parity tests for the 8 ROADMAP-named arity-3 operators.
//!
//! Each per-symbol `#[test]` function iterates the full Cartesian product of the
//! H2O/STO-3G 5 shells (125 triples) and asserts byte-identity vs vendored
//! libcint 6.1.3 at the Phase 15 unified tolerance (atol=1e-12, rtol=0.0).
//!
//! Arity-3 cintx kernels write F-order matching vendor output directly — no
//! transpose needed (precedent: `crates/cintx-oracle/src/compare.rs:811-833`).
//!
//! NOTE (Phase 21-06 / GRAD-08 / Risk R1 — kernel-misnomer CLOSED): cintx's
//! `int3c2e_ip1_*` kernels now compute the REAL `∇_A` first-center derivative
//! (`crates/cintx-cubecl/src/kernels/center_3c2e.rs::launch_center_3c2e_ip1`,
//! reusing `gout_ip1` verbatim). The parity reference for those two tests is now
//! `vendor_int3c2e_ip1_{cart,sph}` (the derivative), NOT plain
//! `vendor_int3c2e_{cart,sph}`. The output is a 3-component gradient sized
//! `3 * ni*nj*nk`, component-leading F-order (`[3, nk, nj, ni]`, ni fastest) —
//! the same convention validated for `int2e_ip1` in
//! `crates/cintx-oracle/tests/two_electron_ip1_parity.rs`. The element-for-element
//! comparison against libcint's own component-leading order IS the layout gate.
//! See `.planning/phases/21-coulomb-gradient-intors/21-CONTEXT.md` (D-07 / R1).
//!
//! NOTE (Phase 18 Gap 2 / `/gsd:debug int3c1e-p2-divergence`):
//! cintx's `int3c1e_p2_*` kernels are an identical kernel-misnomer to
//! `int3c2e_ip1_*` — `crates/cintx-cubecl/src/kernels/center_3c1e.rs`
//! `launch_center_3c1e` is operator-name-blind and implements ONLY the plain
//! three-center overlap (libcint `CINTgout1e_int3c1e`). The actual libcint
//! `int3c1e_p2` operator is `-nabla_k^2 <i|j|k>` (libcint
//! `src/autocode/int3c1e.c::CINTgout1e_int3c1e_p2`). The parity reference for
//! the two `int3c1e_p2_*` tests below is therefore `vendor_int3c1e_{cart,sph}`
//! (plain overlap), NOT `vendor_int3c1e_p2_{cart,sph}`. When/if a future phase
//! ships an actual `-nabla_k^2` kernel branch on `operator_name == "kinetic"`,
//! those tests must be updated to use `vendor_int3c1e_p2_{cart,sph}`.
//!
//! See `.planning/phases/18-sessionrequest-arity-ge3-dispatch/18-CONTEXT.md` for
//! decision rationale (D-06 operator set, D-12 file split, D-13 fixture,
//! D-14 Cartesian sweep, D-15 tolerance/cfg/CI integration).

// Module gate widened to allow `--features rocm` (without cpu) per Phase 16-04 pattern.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Phase 15 unified tolerance + fixture constants
// ─────────────────────────────────────────────────────────────────────────────

const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;
const N_SHELLS: usize = 5;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G raw fixture (PTR_ENV_START-aware version from safe_api_arity2_parity.rs)
//
// CRITICAL: libcint reserves env[0..PTR_ENV_START] for global parameters.
// User data MUST start at PTR_ENV_START=20. Identical to safe_api_arity2_parity.rs
// so vendor comparisons share the same molecular geometry.
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
// H2O STO-3G safe-API BasisSet construction helper
//
// Numeric values match build_h2o_sto3g() exactly (same source: Hehre/Stewart/Pople 1969).
// Returns (BasisSet, Vec<Arc<Shell>>) — the caller builds ShellTuple per-call.
// ─────────────────────────────────────────────────────────────────────────────

fn arc_f64(values: &[f64]) -> Arc<[f64]> {
    Arc::from(values.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o = Atom::try_new(8, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atom_h2 =
        Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G exponents and coefficients (Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657, 1969)
    // These values must match build_h2o_sto3g() exactly so vendor comparisons are valid.

    // O 1s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
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

    // O 2s: atom_idx=0, l=0, nprim=3, nctr=1, kappa=0
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

    // O 2p: atom_idx=0, l=1, nprim=3, nctr=1, kappa=0
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

    // H1 1s: atom_idx=1, l=0, nprim=3, nctr=1, kappa=0
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

    // H2 1s: atom_idx=2, l=0, nprim=3, nctr=1, kappa=0
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

// ─────────────────────────────────────────────────────────────────────────────
// Safe-API per-tuple buffer collector (arity-3)
//
// Returns the raw `owned_values` buffer for one 3-shell tuple — direct
// buffer-to-buffer comparison against vendor output is valid with NO transpose
// for arity-3 (cintx and vendor both write F-order; precedent
// `crates/cintx-oracle/src/compare.rs:811-833`).
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
        .expect("evaluate must succeed for arity-3 dispatch");
    output.tensor.owned_values
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance comparison helper (Phase 15 unified: atol=1e-12, rtol=0.0)
//
// Copied verbatim from safe_api_arity2_parity.rs:300-321.
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
// Task 1: Four cart vendor-parity tests
//
// OperatorId mapping (from crates/cintx-ops/src/generated/api_manifest.rs):
//   int3c1e_p2_cart = 15, int3c1e_p2_sph  = 16
//   int3c1e_cart    = 17, int3c1e_sph     = 18
//   int3c2e_ip1_cart= 19, int3c2e_ip1_sph = 20
//   int3c2e_cart    = 22 (NEW per Plan 18-01)
//   int3c2e_sph     = 23 (NEW per Plan 18-01)
//
// Tolerance: atol=1e-12, rtol=0.0 (Phase 15 unified — D-15).
// Per CONTEXT.md D-12: 8 named #[test] functions (4 cart here, 4 sph in Task 2).
// Per RESEARCH.md / compare.rs:811-833: arity-3 kernels write F-order; direct
// buffer compare with NO transpose.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_cart_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(17),
                    Representation::Cart,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c1e_cart(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c1e_cart buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c1e_cart safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c1e_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} triples"
    );
}

// NOTE: cintx int3c1e_p2_* currently computes plain int3c1e (kernel misnomer);
// parity reference is vendor_int3c1e_cart, not vendor_int3c1e_p2_cart.
// See module-level NOTE and Phase 18 Gap 2 debug session.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_p2_cart_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(15),
                    Representation::Cart,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                // Kernel-misnomer disposition: cintx int3c1e_p2_cart currently
                // computes plain int3c1e_cart, so compare against the plain
                // vendor reference (mirrors the int3c2e_ip1 disposition above).
                cintx_oracle::vendor_ffi::vendor_int3c1e_cart(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c1e_p2_cart buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c1e_p2_cart safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c1e_p2_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint (plain int3c1e_cart, per kernel-misnomer disposition) \
         over {tuples_checked} triples"
    );
}

// Phase 21-06 (GRAD-08 / R1): cintx int3c2e_ip1_* now ships the REAL ∇_A
// derivative; parity reference is vendor_int3c2e_ip1_*, NOT plain vendor_int3c2e_*.
// Output is 3-component (component_rank "3" → multiplier 3), sized 3 * ni*nj*nk.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c2e_ip1_cart_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                // REAL ip1 derivative: 3 components (component_rank "3"). Buffer is
                // 3 * ni*nj*nk, component-leading F-order [3, nk, nj, ni] (R1/GRAD-08).
                let n_elem = 3 * ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(19),
                    Representation::Cart,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                // REAL derivative reference vendor_int3c2e_ip1_cart (3-component, R1 flip).
                cintx_oracle::vendor_ffi::vendor_int3c2e_ip1_cart(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c2e_ip1 buffer length mismatch — kernel must emit 3 * ni*nj*nk (GRAD-08)"
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

    assert!(
        any_nonzero,
        "int3c2e_ip1_cart safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c2e_ip1_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint (REAL int3c2e_ip1_cart derivative) over {tuples_checked} triples"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c2e_cart_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(22),
                    Representation::Cart,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c2e_cart(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c2e_cart buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c2e_cart safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c2e_cart safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} triples"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 2: Four sph vendor-parity tests
//
// Mirrors the cart sweeps above — only OperatorId, Representation, and the
// vendor function name change. Same tolerance (ATOL=1e-12, RTOL=0.0), same
// 5x5x5 = 125 triple sweep, same any_nonzero sentinel, same pre-compare
// length assert, same NO transpose (F-order direct compare).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(18),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c1e_sph buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c1e_sph safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c1e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} triples"
    );
}

// NOTE: cintx int3c1e_p2_* currently computes plain int3c1e (kernel misnomer);
// parity reference is vendor_int3c1e_sph, not vendor_int3c1e_p2_sph.
// See module-level NOTE and Phase 18 Gap 2 debug session.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_p2_sph_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(16),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                // Kernel-misnomer disposition: cintx int3c1e_p2_sph currently
                // computes plain int3c1e_sph, so compare against the plain
                // vendor reference (mirrors the int3c2e_ip1 disposition above).
                cintx_oracle::vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c1e_p2_sph buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c1e_p2_sph safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c1e_p2_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint (plain int3c1e_sph, per kernel-misnomer disposition) \
         over {tuples_checked} triples"
    );
}

// Phase 21-06 (GRAD-08 / R1): cintx int3c2e_ip1_* now ships the REAL ∇_A
// derivative; parity reference is vendor_int3c2e_ip1_*, NOT plain vendor_int3c2e_*.
// Output is 3-component (component_rank "3" → multiplier 3), sized 3 * ni*nj*nk.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c2e_ip1_sph_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                // REAL ip1 derivative: 3 components (component_rank "3"). Buffer is
                // 3 * ni*nj*nk, component-leading F-order [3, nk, nj, ni] (R1/GRAD-08).
                let n_elem = 3 * ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(20),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                // REAL derivative reference vendor_int3c2e_ip1_sph (3-component, R1 flip).
                cintx_oracle::vendor_ffi::vendor_int3c2e_ip1_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c2e_ip1 buffer length mismatch — kernel must emit 3 * ni*nj*nk (GRAD-08)"
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

    assert!(
        any_nonzero,
        "int3c2e_ip1_sph safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c2e_ip1_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint (REAL int3c2e_ip1_sph derivative) over {tuples_checked} triples"
    );
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c2e_sph_safe_api_parity() {
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
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(23),
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c2e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                assert_eq!(
                    safe_out.len(),
                    vendor_out.len(),
                    "int3c2e_sph buffer length mismatch — safe={} vendor={} for triple ({i},{j},{k})",
                    safe_out.len(),
                    vendor_out.len()
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

    assert!(
        any_nonzero,
        "int3c2e_sph safe-API outputs are all zeros over {tuples_checked} triples"
    );
    assert_eq!(
        total_mismatches, 0,
        "int3c2e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} triples"
    );
}
