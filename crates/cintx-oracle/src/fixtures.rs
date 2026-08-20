use anyhow::{Context, Result, anyhow, bail};
use cintx_compat::helpers::{CINTcgto_cart, CINTcgto_spheric, CINTcgto_spinor};
use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NGRIDS, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_F12_ZETA,
    PTR_GRIDS, PTR_ZETA,
};
use cintx_core::Representation;
use cintx_ops::resolver::{HelperKind, ManifestEntry, Resolver};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G molecular fixture (PTR_ENV_START-aligned)
// ─────────────────────────────────────────────────────────────────────────────

/// Build H2O STO-3G libcint-style atm/bas/env with user data starting at PTR_ENV_START.
///
/// PTR_ENV_START alignment is required for 2e-family integrals (2c2e, 3c2e, 2e)
/// to avoid corrupting libcint global env slots (e.g., PTR_RANGE_OMEGA at index 8).
///
/// Molecule: H2O with O at origin, H1 at (0, 1.4307, 1.1078) Bohr, H2 at (0, -1.4307, 1.1078) Bohr.
/// Basis: STO-3G (Hehre, Stewart & Pople, JCP 51, 2657, 1969).
/// Shells: 0=O-1s, 1=O-2s, 2=O-2p, 3=H1-1s, 4=H2-1s.
pub fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    // env[0..PTR_ENV_START] reserved for libcint global params (zeros = defaults).
    let mut env = vec![0.0_f64; PTR_ENV_START];

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

    // atm: O, H1, H2
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

    // bas: O-1s, O-2s, O-2p, H1-1s, H2-1s
    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

/// Build H2O STO-3G fixture with PTR_F12_ZETA set for F12 oracle parity tests.
///
/// Sets `env[PTR_F12_ZETA]` (env[9]) to the given `zeta` value. This is required
/// for all F12/STG/YP integrals. A zeta of 0.0 must be explicitly rejected by the
/// cintx engine via `InvalidEnvParam`.
///
/// Typical value: `zeta = 1.2` (common F12 correlation factor exponent in production).
pub fn build_h2o_sto3g_f12(zeta: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    // PTR_F12_ZETA = 9 — within the PTR_ENV_START global params block.
    env[PTR_F12_ZETA] = zeta;
    (atm, bas, env)
}

/// Default NON-ZERO gauge origin for the Phase 22 fixture (Bohr).
/// Non-trivial on all three axes so a populated `common_orig` is distinguishable
/// from the `[0,0,0]` default (CONTEXT line 103: a zero origin proves nothing).
pub const COMMON_ORIG_FIXTURE_ORIGIN: [f64; 3] = [0.5, -0.3, 0.8];

/// H2O/STO-3G fixture with a NON-ZERO gauge origin set at env[PTR_COMMON_ORIG..+3].
///
/// Data infrastructure for moment (Phase 24) / GIAO (Phase 26) byte-identity parity.
/// Phase 22 (FND-01) only builds + round-trips this fixture; no consuming kernel exists yet (D-03).
pub fn build_h2o_sto3g_common_orig() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    build_h2o_sto3g_common_orig_at(COMMON_ORIG_FIXTURE_ORIGIN)
}

/// H2O/STO-3G fixture with an explicit gauge origin set at env[PTR_COMMON_ORIG..+3].
pub fn build_h2o_sto3g_common_orig_at(origin: [f64; 3]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    // PTR_COMMON_ORIG = 1 — gauge origin in the PTR_ENV_START global params block.
    env[PTR_COMMON_ORIG] = origin[0];
    env[PTR_COMMON_ORIG + 1] = origin[1];
    env[PTR_COMMON_ORIG + 2] = origin[2];
    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// D-08 ADVERSARIAL spinor-derivative fixture (Phase 27 Plan 01 Task 2).
//
// A single fixture that simultaneously trips the four distinct silent-false-pass
// landmines for spinor derivative parity (see 27-SPIKE-FINDINGS.md / threat
// register T-27-02):
//
//   1. NON-SQUARE bra/ket (p × d): di != dj, so an i/j-transposed misread of the
//      derivative cart block can no longer pass (the square-block H2O test was
//      orientation-blind).
//   2. nctr>1 on the bra (p, nctr=2): forces the general-contraction composition
//      `i_global = ci*di + ic` and the COLUMN-major env coeff (env[ci*nprim+ip])
//      → ROW-major Shell transpose. A dropped contraction column or a transposed
//      coeff layout diverges.
//   3. kappa=0 on EVERY bas row: both the j=l+1/2 (GT) and j=l-1/2 (LT) spinor
//      blocks fire and the spinor axis length is `CINTcgto_spinor = 4l+2`. A
//      half-block (single-j) sizing diverges.
//   4. NON-ZERO rinv origin (env[PTR_RINV_ORIG..+3]): the int3c1e_iprinv /
//      int1e_*rinv paths read the rinv center; a zero-origin shortcut would
//      trivially pass, so the origin is displaced (Phase 24/25 landmine).
//
// Shell triple (the third k shell is the auxiliary center for the arity-3
// int3c1e/int3c2e families):
//   shell 0: bra i  = p-shell (l=1), 3 primitives, nctr_i = 2  (gc, two columns)
//   shell 1: ket j  = d-shell (l=2), 3 primitives, nctr_j = 1
//   shell 2: aux k  = s-shell (l=0), 3 primitives, nctr_k = 1
//
// Modeled VERBATIM on `int3c1e_genctr_parity.rs::build_genctr_fixture`, with the
// added kappa=0 spinor setup (KAPPA_OF=0 on every row) from
// `one_electron_grad_spinor_parity.rs::build_h2o_sto3g_spinor`.
// ─────────────────────────────────────────────────────────────────────────────

/// Build the D-08 adversarial spinor-derivative fixture (atm, bas, env).
///
/// Non-square (p × d) + nctr>1 (p has nctr=2) + kappa=0 (every bas row) +
/// non-zero rinv origin (env[PTR_RINV_ORIG..+3]). Exposes a third aux-k (s) shell
/// for the arity-3 int3c1e/int3c2e spinor derivative families.
pub fn build_adversarial_spinor_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::{KAPPA_OF, PTR_RINV_ORIG};

    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];
    let k_coord = [0.9_f64, -0.4, 0.2];

    // p-shell (bra i) — 3 primitives, two general-contraction columns.
    // The libcint env coefficient block is COLUMN-MAJOR: env[ci*nprim + ip]
    // (CINTprim_to_ctr_0 in g1e.c). cintx transposes it to row-major internally.
    //   column 0 = (0.70, 0.30, 0.15) , column 1 = (0.20, 0.55, 0.80)
    // → env layout [c0_p0, c0_p1, c0_p2, c1_p0, c1_p1, c1_p2].
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];

    // d-shell (ket j) — 3 primitives, single contraction.
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    // s-shell (aux k) — 3 primitives, single contraction.
    let s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // Reserve the libcint global-parameter region; rinv origin lives in
    // env[PTR_RINV_ORIG..+3]. Use a PTR_ENV_START-aligned reserve so 2e-family
    // global slots are never clobbered.
    let mut env = vec![0.0_f64; PTR_ENV_START];
    // NON-ZERO rinv origin — exercises the iprinv center path (T-27-04).
    env[PTR_RINV_ORIG] = 0.30;
    env[PTR_RINV_ORIG + 1] = -0.45;
    env[PTR_RINV_ORIG + 2] = 0.60;

    let i_coord_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_coord_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
    let k_coord_ptr = env.len() as i32;
    env.extend_from_slice(&k_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for (n, &ptr) in [i_coord_ptr, j_coord_ptr, k_coord_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    // shell 0: p, nctr=2 (general contraction), spinor (kappa=0)
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
    bas[KAPPA_OF] = 0;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: d, nctr=1, spinor (kappa=0)
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = 0;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;
    // shell 2: s (aux k), nctr=1, spinor (kappa=0)
    bas[2 * BAS_SLOTS + ATOM_OF] = 2;
    bas[2 * BAS_SLOTS + ANG_OF] = 0;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[2 * BAS_SLOTS + PTR_EXP] = s_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = s_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 28 (FND-05 / D-05): kappa≠0 adversarial spinor fixture.
//
// PRIMARY GATE. Clones build_adversarial_spinor_fixture's geometry — NON-SQUARE
// (p × d) bra/ket with a general-contracted (nctr=2) bra, distinct centers — but
// sets GENUINE kappa≠0 so the si transform is proven on the non-(4l+2) spinor
// sizing path that the kappa=0 fixture structurally cannot exercise:
//   p shell  kappa = +1  → LT-only j=l−1/2, di = spinor_len(1, +1) = 2*1     = 2
//   d shell  kappa = −1  → GT-only j=l+1/2, dj = spinor_len(2, −1) = 2*2 + 2 = 6
// The block is 2×6 (non-square) and the staging buffer is di*dj*2 = 24 f64. This
// is the FIRST cintx fixture exercising the GT/LT-only sizing (Spike Target D).
// Two shells only (no aux-k) — the σ·p int1e_sp vehicle is arity-2.
// ─────────────────────────────────────────────────────────────────────────────

/// Build the D-05 PRIMARY kappa≠0 adversarial spinor fixture (atm, bas, env).
///
/// Non-square (p × d) + nctr>1 (p has nctr=2) + GENUINE kappa≠0 (p kappa=+1 LT,
/// d kappa=−1 GT) so the si_2d transform runs on the non-`(4l+2)` spinor sizing
/// path (`di = 2l`, `dj = 2l+2`). Mirrors `build_adversarial_spinor_fixture`'s
/// geometry/coeffs; only KAPPA_OF changes (and the aux-k shell is dropped — the
/// int1e_sp σ·p vehicle is arity-2).
pub fn build_kappa_spinor_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::{KAPPA_OF, PTR_RINV_ORIG};

    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];

    // p-shell (bra i) — 3 primitives, two general-contraction columns. ROW-major
    // here; libcint env is COLUMN-major env[ci*nprim + ip] (cintx transposes
    // internally — project_raw_nctr_coeff_transpose). column 0 / column 1:
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];

    // d-shell (ket j) — 3 primitives, single contraction.
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let mut env = vec![0.0_f64; PTR_ENV_START];
    // NON-ZERO rinv origin retained (harmless for int1e_sp; keeps the geometry
    // identical to the adversarial template).
    env[PTR_RINV_ORIG] = 0.30;
    env[PTR_RINV_ORIG + 1] = -0.45;
    env[PTR_RINV_ORIG + 2] = 0.60;

    let i_coord_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_coord_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (n, &ptr) in [i_coord_ptr, j_coord_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    // shell 0: p, nctr=2 (general contraction), kappa=+1 → LT-only (di = 2*1 = 2).
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
    bas[KAPPA_OF] = 1; // p kappa = +1 (LT)
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: d, nctr=1, kappa=−1 → GT-only (dj = 2*2 + 2 = 6).
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = -1; // d kappa = −1 (GT)
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 30 (GIAO-03 / D-02): combined gauge≠0 ∧ kappa≠0 1e spinor fixture.
//
// PRIMARY Wave-0/Wave-1 GATE for the GIAO×σ 1e families. Internally reproduces
// build_kappa_spinor_fixture's exact geometry (p kappa=+1 LT nctr=2 × d kappa=−1
// GT, non-square (2,6), COLUMN-major env coeffs) and ADDITIONALLY sets a non-zero
// asymmetric gauge origin at env[PTR_COMMON_ORIG..+3]. The `int1e_cg_sa10*` arms
// read PTR_COMMON_ORIG (dri = ri − common_orig); without a non-zero origin the
// gauge×kappa cross-term the integrand couples is never exercised.
//
// The 5 D-02 hard constraints (all hold):
//   (1) env[PTR_COMMON_ORIG..+3] = [0.30, -0.45, 0.60] ≠ [0,0,0]  — cg_sa10* reads it
//   (2) kappa≠0 GT/LT mix: p kappa=+1 (LT, di=2) AND d kappa=−1 (GT, dj=6) →
//       BOTH spinor_len branches
//   (3) non-square block dims (2,6) — defeats KET→BRA transpose symmetry
//   (4) ≥1 shell with nctr>1: the p shell keeps NCTR_OF=2 (COLUMN-major env coeff)
//   (5) the PTR_RINV_ORIG block stays for the nucsp/sa01 Rys arms
//
// Does NOT mutate build_kappa_spinor_fixture (Phase-29 REL tests depend on it).
// ─────────────────────────────────────────────────────────────────────────────

/// Build the D-02 combined gauge≠0 ∧ kappa≠0 **1e** spinor fixture (atm, bas, env).
///
/// Same geometry as [`build_kappa_spinor_fixture`] (p kappa=+1 LT nctr=2 × d
/// kappa=−1 GT, non-square (2,6)) plus a non-zero asymmetric gauge origin written
/// at `env[PTR_COMMON_ORIG..+3]`. This is the Wave-0 gauge-gout micro-test vehicle
/// and the Wave-1 1e parity gate for every GIAO×σ 1e family.
pub fn build_gauge_kappa_spinor_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::PTR_COMMON_ORIG;

    // Reuse the Phase-29 kappa fixture verbatim (geometry, coeffs, kappa mix,
    // nctr>1, PTR_RINV_ORIG). The ONLY D-02 addition is the gauge origin below.
    let (atm, bas, mut env) = build_kappa_spinor_fixture();

    // D-02 constraint (1): non-zero asymmetric gauge origin — the cg_sa10* arms
    // read dri = ri − env[PTR_COMMON_ORIG]. PTR_COMMON_ORIG = 1 (raw.rs:50). Same
    // off-center asymmetric triple already used for PTR_RINV_ORIG so the gauge≠0
    // path is exercised without re-tuning convergence.
    env[PTR_COMMON_ORIG] = 0.30;
    env[PTR_COMMON_ORIG + 1] = -0.45;
    env[PTR_COMMON_ORIG + 2] = 0.60;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 29 (REL-03/04 / D-02): kappa≠0 adversarial 2e (4-shell) spinor fixture.
//
// PRIMARY GATE for every REL-03/04 2e σ family. A 2-electron quartet (i,j,k,l) of
// four spinor shells, extending build_kappa_spinor_fixture's geometry/coeffs to four
// distinct centers. HARD D-02 constraints, all satisfied:
//   - exactly 4 spinor shells (a 2-electron quartet)
//   - NON-SQUARE: the four spinor dims (di,dj,dk,dl) are all distinct (2,6,2,4) so
//     transpose/orientation symmetry that hides a KET→BRA omission is defeated
//   - GENUINE kappa≠0 GT/LT mix: shell i is kappa=+1 (LT, 2l); the other three are
//     kappa=−1 (GT, 2l+2) → exercises BOTH spinor_len sizing branches, not just 4l+2
//   - ≥1 shell with nctr>1: shell i has NCTR_OF=2 with the COLUMN-major env coeff
//     layout [c0_p0,c0_p1,c0_p2, c1_p0,c1_p1,c1_p2] (catches the coeff transpose)
//
// Spinor dims: i = p kappa=+1 → di = spinor_len(1,+1)  = 2*1     = 2  (LT)
//              j = d kappa=−1 → dj = spinor_len(2,−1)  = 2*2 + 2 = 6  (GT)
//              k = s kappa=−1 → dk = spinor_len(0,−1)  = 2*0 + 2 = 2  (GT)
//              l = p kappa=−1 → dl = spinor_len(1,−1)  = 2*1 + 2 = 4  (GT)
// The four-shell quartet rides distinct centers (a quartet of light atoms) so the
// 2e cross blocks are genuinely non-zero. build_heavy_atom_spinor_fixture remains the
// secondary realism cross-check (asserted finite, NOT the primary gate).
// ─────────────────────────────────────────────────────────────────────────────

/// Build the D-02 PRIMARY kappa≠0 adversarial **2e** (4-shell) spinor fixture
/// (atm, bas, env).
///
/// Four spinor shells forming a 2-electron quartet `(i,j,k,l)`:
///   - i: p, kappa=+1 (LT, di=2), **nctr=2** (general contraction, column-major coeff)
///   - j: d, kappa=−1 (GT, dj=6), nctr=1
///   - k: s, kappa=−1 (GT, dk=2), nctr=1
///   - l: p, kappa=−1 (GT, dl=4), nctr=1
///
/// Non-square (2,6,2,4) + GT/LT kappa mix + nctr>1 — the byte-identity gate for the
/// REL-03/04 2e σ families (proven byte-identical by the 29-04 micro-test). NEVER use a
/// square quartet (orientation/transpose bugs hide). Exact element/kappa assignment is
/// Claude's discretion subject to the D-02 hard constraints.
pub fn build_kappa_spinor_2e_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::{KAPPA_OF, PTR_RINV_ORIG};

    // Four distinct centers (a light-atom quartet — non-zero 2e cross blocks).
    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];
    let k_coord = [0.9_f64, -0.4, 0.2];
    let l_coord = [-0.6_f64, 0.5, -0.8];

    // i p-shell — 3 primitives, two general-contraction columns. COLUMN-major env
    // layout env[ci*nprim + ip] (cintx transposes to row-major internally —
    // project_raw_nctr_coeff_transpose): column 0 = (0.70,0.30,0.15),
    // column 1 = (0.20,0.55,0.80).
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];

    // j d-shell — 3 primitives, single contraction.
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    // k s-shell — 3 primitives, single contraction.
    let s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // l p-shell — 3 primitives, single contraction (distinct exponents from i).
    let pl_exp = [2.9412494_f64, 0.6834831, 0.2222899];
    let pl_coeff = [0.62391373_f64, 0.42179344, 0.11000000];

    let mut env = vec![0.0_f64; PTR_ENV_START];
    // NON-ZERO rinv origin retained (harmless for the overlap-style 2e σ vehicles;
    // keeps the geometry consistent with the 1e adversarial template).
    env[PTR_RINV_ORIG] = 0.30;
    env[PTR_RINV_ORIG + 1] = -0.45;
    env[PTR_RINV_ORIG + 2] = 0.60;

    let i_coord_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_coord_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
    let k_coord_ptr = env.len() as i32;
    env.extend_from_slice(&k_coord);
    let l_coord_ptr = env.len() as i32;
    env.extend_from_slice(&l_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let pl_exp_ptr = env.len() as i32;
    env.extend_from_slice(&pl_exp);
    let pl_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&pl_coeff);

    let mut atm = vec![0_i32; 4 * ATM_SLOTS];
    for (n, &ptr) in [i_coord_ptr, j_coord_ptr, k_coord_ptr, l_coord_ptr]
        .iter()
        .enumerate()
    {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 4 * BAS_SLOTS];
    // shell 0 (i): p, nctr=2 (general contraction), kappa=+1 → LT-only (di = 2*1 = 2).
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
    bas[KAPPA_OF] = 1; // p kappa = +1 (LT)
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1 (j): d, nctr=1, kappa=−1 → GT-only (dj = 2*2 + 2 = 6).
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = -1; // d kappa = −1 (GT)
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;
    // shell 2 (k): s, nctr=1, kappa=−1 → GT-only (dk = 2*0 + 2 = 2).
    bas[2 * BAS_SLOTS + ATOM_OF] = 2;
    bas[2 * BAS_SLOTS + ANG_OF] = 0;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + KAPPA_OF] = -1; // s kappa = −1 (GT)
    bas[2 * BAS_SLOTS + PTR_EXP] = s_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = s_coeff_ptr;
    // shell 3 (l): p, nctr=1, kappa=−1 → GT-only (dl = 2*1 + 2 = 4).
    bas[3 * BAS_SLOTS + ATOM_OF] = 3;
    bas[3 * BAS_SLOTS + ANG_OF] = 1;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + KAPPA_OF] = -1; // p kappa = −1 (GT)
    bas[3 * BAS_SLOTS + PTR_EXP] = pl_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = pl_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 28 (FND-05 / D-05): heavy-atom realism cross-check fixture.
//
// SECONDARY (non-primary) gate. A small heavy-element relativistic 2c spinor basis
// (a Dirac/dyall-flavored p×d pair) guards against synthetic-fixture blind spots in
// the adversarial kappa fixture. Genuine kappa≠0 (p kappa=+1 LT, d kappa=−1 GT) so
// the realism check also rides the non-(4l+2) sizing path. Element/exponents are
// Claude's discretion (D-05 / A3).
//
// The two shells sit on DISTINCT centers (a heavy-atom HYDRIDE-style environment:
// Hg + a displaced ligand center). A same-center σ·p p×d block vanishes by selection
// rules (an int1e_sp ⟨p|σ·p|d⟩ on one spherically-symmetric center is ~0); a real
// molecular environment is the physically meaningful realism check and yields a
// genuinely non-zero cross block — exactly the synthetic-blind-spot guard D-05 wants.
// ─────────────────────────────────────────────────────────────────────────────

/// Build the D-05 SECONDARY heavy-atom realism cross-check fixture (atm, bas, env).
///
/// Hg (Z=80) p (kappa=+1, LT) shell + a displaced ligand-center d (kappa=−1, GT)
/// shell, nctr=1 — a heavy-atom hydride-style 2-center environment. Realism
/// cross-check for the synthetic kappa fixture — NOT the primary byte-identity gate.
/// Distinct centers so the σ·p p×d cross block is genuinely non-zero (a same-center
/// block vanishes by selection rules).
pub fn build_heavy_atom_spinor_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    use cintx_compat::raw::KAPPA_OF;

    // Heavy center (Hg, Z=80) at the origin; a displaced ligand center for the d shell.
    let hg_center = [0.0_f64, 0.0, 0.0];
    let lig_center = [0.0_f64, 0.0, 1.62]; // ~Hg–H bond length scale (Bohr)

    // Heavy-atom-flavored exponents (tight + diffuse), single contraction each.
    let p_exp = [12.5_f64, 2.85, 0.74];
    let p_coeff = [0.21_f64, 0.55, 0.34];
    let d_exp = [9.30_f64, 2.10, 0.52];
    let d_coeff = [0.18_f64, 0.58, 0.37];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let hg_ptr = env.len() as i32;
    env.extend_from_slice(&hg_center);
    let lig_ptr = env.len() as i32;
    env.extend_from_slice(&lig_center);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 80; // Hg
    atm[PTR_COORD] = hg_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1; // ligand (H)
    atm[ATM_SLOTS + PTR_COORD] = lig_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    // shell 0: Hg p, nctr=1, kappa=+1 → LT (di = 2).
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[KAPPA_OF] = 1;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: ligand-center d, nctr=1, kappa=−1 → GT (dj = 6).
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + KAPPA_OF] = -1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    (atm, bas, env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cu/LANL2DZ molecular fixture (Phase 19 Plan 01 Wave 0 scaffold).
//
// Source: basissetexchange.org "LANL2DZ" element 29 (Cu); see
// `crates/cintx-oracle/data/cu_lanl2dz.json` for the embedded basis +
// ECP parameters with provenance.
// Original paper: Hay & Wadt, J. Chem. Phys. 82, 270 (1985).
// AO contractions: 3 (s) + 3 (p) + 2 (d) = 8 shells (general contractions
// from BSE split into single-NCTR libcint rows). ECP projectors: 3
// (l=0 s projector, l=1 p projector, l=2 d/local channel).
// ─────────────────────────────────────────────────────────────────────────────

const CU_LANL2DZ_JSON: &str = include_str!("../data/cu_lanl2dz.json");

/// Parsed Cu/LANL2DZ basis JSON, populated once on first call.
fn cu_lanl2dz_parsed() -> &'static Value {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Value> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(CU_LANL2DZ_JSON)
            .expect("cu_lanl2dz.json must parse as valid JSON at compile-baked time")
    })
}

/// Build Cu/LANL2DZ libcint-style (atm, bas, ecpbas, env) slabs with user
/// data starting at PTR_ENV_START and `env[AS_ECPBAS_OFFSET]` /
/// `env[AS_NECPBAS]` populated per PySCF `nr_ecp.h` convention.
///
/// The returned `ecpbas` slab is a separate `Vec<i32>` of width
/// `BAS_SLOTS = 8` per row (slots 3 and 4 reinterpreted as `RADI_POWER` and
/// `SO_TYPE_OF` for ECP rows per Phase 19 D-05). Callers either pass the
/// `ecpbas` pointer to the FFI separately, or pack it into a combined slab
/// and update `env[AS_ECPBAS_OFFSET]` to point at it; this Wave 0 stub
/// keeps the slabs separate and sets `env[AS_ECPBAS_OFFSET] = 0` (sentinel
/// for "ecpbas passed as a separate slab"). `env[AS_NECPBAS]` is set to
/// the row count for forward compatibility.
///
/// PySCF nr_ecp.h slot constants (mirrored locally for self-containment):
///   AS_ECPBAS_OFFSET = 18, AS_NECPBAS = 19, RADI_POWER = 3, SO_TYPE_OF = 4.
///
/// Per Phase 19 D-04 (channel encoding): the ECP "local" channel is stored
/// with `ANG_OF = -1` (PySCF convention). The d-channel (l=2) is the local
/// channel in LANL2DZ for Cu; s (l=0) and p (l=1) are semi-local projectors.
///
/// Plan 03 promotes this scaffold to consume the typed `EcpShell` /
/// `BasisSet::ecp_shells` surface and the typed safe-API entry point.
pub fn build_cu_lanl2dz() -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>) {
    // PySCF nr_ecp.h slot constants — duplicated here for clarity. These
    // match `vendor/pyscf-nr-ecp/include/nr_ecp.h` and will be promoted to
    // `cintx-compat::raw` constants in Plan 03.
    const AS_ECPBAS_OFFSET: usize = 18;
    const AS_NECPBAS: usize = 19;
    const RADI_POWER_SLOT: usize = 3;
    const SO_TYPE_OF_SLOT: usize = 4;

    let parsed = cu_lanl2dz_parsed();

    // ─── Read atom data ──────────────────────────────────────────────────
    let atom_z = parsed["atom"]["Z"]
        .as_i64()
        .expect("cu_lanl2dz.json: atom.Z must be integer") as i32;
    let coord_arr = parsed["atom"]["coord"]
        .as_array()
        .expect("cu_lanl2dz.json: atom.coord must be array");
    let cu_coord: [f64; 3] = [
        coord_arr[0].as_f64().unwrap(),
        coord_arr[1].as_f64().unwrap(),
        coord_arr[2].as_f64().unwrap(),
    ];

    // ─── Build env with PTR_ENV_START prepad ─────────────────────────────
    let mut env = vec![0.0_f64; PTR_ENV_START];

    let cu_coord_ptr = env.len() as i32; // typically 20
    env.extend_from_slice(&cu_coord);
    let zeta_ptr = env.len() as i32; // 23 — nuclear model zeta slot (unused for POINT_NUC)
    env.push(0.0);

    // Append AO shell exponents + coefficients.
    let shells_json = parsed["shells"]
        .as_array()
        .expect("cu_lanl2dz.json: shells must be array");
    let mut shell_entries: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(shells_json.len()); // (l, nprim, exp_ptr, coeff_ptr)
    for shell in shells_json {
        let l = shell["l"]
            .as_i64()
            .expect("cu_lanl2dz.json: shell.l must be integer") as i32;
        let exps: Vec<f64> = shell["exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let coeffs: Vec<f64> = shell["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        debug_assert_eq!(
            exps.len(),
            coeffs.len(),
            "Cu/LANL2DZ shell: nprim mismatch between exponents and coefficients"
        );
        let nprim = exps.len() as i32;
        let exp_ptr = env.len() as i32;
        env.extend_from_slice(&exps);
        let coeff_ptr = env.len() as i32;
        env.extend_from_slice(&coeffs);
        shell_entries.push((l, nprim, exp_ptr, coeff_ptr));
    }

    // Append ECP shell exponents + coefficients.
    let ecp_shells_json = parsed["ecp"]["shells"]
        .as_array()
        .expect("cu_lanl2dz.json: ecp.shells must be array");
    let mut ecp_entries: Vec<(i32, i32, i32, i32, i32, i32)> =
        Vec::with_capacity(ecp_shells_json.len());
    // (ang_of, radial_power_sum, so_type, nprim, exp_ptr, coeff_ptr).
    // radial_power_sum is the FIRST r_exponent in the list (PySCF stores a
    // single integer at the RADI_POWER slot; per-primitive r_exponents in
    // the LANL2DZ JSON share the same value or are uniform per shell;
    // Plan 03 may split a shell into multiple rows if r_exponents differ).
    for shell in ecp_shells_json {
        let channel = shell["channel"]
            .as_str()
            .expect("cu_lanl2dz.json: ecp.shells[].channel must be string");
        let ang_of: i32 = if channel == "local" {
            -1
        } else {
            shell["l"].as_i64().unwrap() as i32
        };
        let r_exps: Vec<i32> = shell["r_exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let radial_power = if r_exps.is_empty() {
            0
        } else {
            // For Wave 0 we store the first r_exponent; Plan 03/04 split
            // shells whose r_exponents are non-uniform.
            r_exps[0]
        };
        let so_type = 0_i32; // scalar ECP per D-12 (no SO this phase)
        let exps: Vec<f64> = shell["exponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let coeffs: Vec<f64> = shell["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        debug_assert_eq!(
            exps.len(),
            coeffs.len(),
            "Cu/LANL2DZ ECP shell: nprim mismatch between exponents and coefficients"
        );
        let nprim = exps.len() as i32;
        let exp_ptr = env.len() as i32;
        env.extend_from_slice(&exps);
        let coeff_ptr = env.len() as i32;
        env.extend_from_slice(&coeffs);
        ecp_entries.push((ang_of, radial_power, so_type, nprim, exp_ptr, coeff_ptr));
    }

    // ─── Build atm ───────────────────────────────────────────────────────
    let mut atm = vec![0_i32; ATM_SLOTS];
    atm[CHARGE_OF] = atom_z;
    atm[PTR_COORD] = cu_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;

    // ─── Build bas (one row per AO shell, NCTR_OF = 1) ───────────────────
    let mut bas = vec![0_i32; shell_entries.len() * BAS_SLOTS];
    for (idx, (l, nprim, exp_ptr, coeff_ptr)) in shell_entries.iter().enumerate() {
        let base = idx * BAS_SLOTS;
        bas[base + ATOM_OF] = 0;
        bas[base + ANG_OF] = *l;
        bas[base + NPRIM_OF] = *nprim;
        bas[base + NCTR_OF] = 1;
        bas[base + PTR_EXP] = *exp_ptr;
        bas[base + PTR_COEFF] = *coeff_ptr;
    }

    // ─── Build ecpbas (one row per ECP projector, width = BAS_SLOTS) ─────
    let mut ecpbas = vec![0_i32; ecp_entries.len() * BAS_SLOTS];
    for (idx, (ang_of, radial_power, so_type, nprim, exp_ptr, coeff_ptr)) in
        ecp_entries.iter().enumerate()
    {
        let base = idx * BAS_SLOTS;
        ecpbas[base + ATOM_OF] = 0;
        ecpbas[base + ANG_OF] = *ang_of;
        ecpbas[base + NPRIM_OF] = *nprim;
        ecpbas[base + RADI_POWER_SLOT] = *radial_power;
        ecpbas[base + SO_TYPE_OF_SLOT] = *so_type;
        ecpbas[base + PTR_EXP] = *exp_ptr;
        ecpbas[base + PTR_COEFF] = *coeff_ptr;
    }

    // Wire env[AS_ECPBAS_OFFSET] / env[AS_NECPBAS]. For Wave 0 the ecpbas
    // slab is returned separately; we set AS_ECPBAS_OFFSET = 0 as a
    // sentinel meaning "ecpbas is not packed into a combined env-anchored
    // slab" (the FFI wrapper Plan 03 adds will pass the ecpbas pointer
    // alongside atm/bas). AS_NECPBAS is set to the row count so downstream
    // consumers can sanity-check the slab length.
    env[AS_ECPBAS_OFFSET] = 0.0;
    env[AS_NECPBAS] = ecp_entries.len() as f64;

    (atm, bas, ecpbas, env)
}

pub const REQUIRED_MATRIX_ARTIFACT: &str =
    "/tmp/cintx_artifacts/cintx_phase_04_manifest_representation_matrix.json";
pub const MATRIX_ARTIFACT_FALLBACK_NAME: &str =
    "cintx_phase_04_manifest_representation_matrix.json";
pub const REQUIRED_REPORT_ARTIFACT: &str =
    "/tmp/cintx_artifacts/cintx_phase_04_compat_parity_report.json";
pub const REPORT_ARTIFACT_FALLBACK_NAME: &str = "cintx_phase_04_compat_parity_report.json";
pub const PHASE4_APPROVED_PROFILES: &[&str] =
    &["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"];
const ORACLE_COMPARE_APPROVED_PROFILES: &[&str] = &[
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e",
    "unstable-source",
];
pub const PHASE4_ORACLE_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e", "4c1e"];
pub const PHASE2_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e"];

/// Canonical families that are oracle-covered, but verified by a dedicated harness
/// rather than the generic raw-eval/legacy-wrapper parity matrix built here.
///
/// `ecp` is structurally incompatible with the generic `RawApiId` raw-eval path: it
/// requires a separate `ecpbas` slab plus the family-level `launch_ecp` dispatcher, so
/// it has no `RawApiId` mapping in `compare::raw_api_for_symbol`. Its byte-identity
/// against vendored PySCF `nr_ecp` (atol=1e-12) is fully verified in the dedicated
/// harness `crates/cintx-oracle/tests/safe_api_ecp_parity.rs`. These entries therefore
/// keep `oracle_covered=true` in the manifest lock, but are excluded from the generic
/// representation matrix and its expected-symbol completeness check below.
///
/// Matched against `ManifestEntry::canonical_family` / the lock's `canonical_family`
/// field (NOT `id.family`, which is `"1e"` for ECP). Add future dedicated-harness
/// canonical families here so both the matrix source and the expected-symbol set stay
/// consistent from a single source of truth.
const DEDICATED_ORACLE_FAMILIES: &[&str] = &["ecp"];

/// True when `canonical_family` is verified by a dedicated harness and must be excluded
/// from the generic representation matrix and its completeness check.
///
/// Public so the xtask `manifest-audit --check-lock` gate can apply the SAME exclusion
/// to its lock-side symbol collection — otherwise the generated matrix (which excludes
/// ECP) and the lock symbol set (which would include ECP) disagree and the drift gate
/// false-positives on the dedicated-harness symbols.
pub fn is_dedicated_oracle_family(canonical_family: &str) -> bool {
    DEDICATED_ORACLE_FAMILIES.contains(&canonical_family)
}
pub const COMPILED_MANIFEST_LOCK_JSON: &str =
    include_str!("../../cintx-ops/generated/compiled_manifest.lock.json");
const BASE_PROFILE: &str = "base";
const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const FALLBACK_ARTIFACT_DIR_DEFAULT: &str = "/tmp/cintx_artifacts";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleFixture {
    pub family: String,
    pub symbol: String,
    pub representation: String,
    pub arity: usize,
    pub dims: Vec<usize>,
    pub component_count: usize,
    pub complex_interleaved: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileRepresentationMatrix {
    pub profile: String,
    pub fixtures: Vec<OracleFixture>,
}

impl OracleFixture {
    pub fn required_elements(&self) -> usize {
        let base = self
            .dims
            .iter()
            .fold(self.component_count.max(1), |acc, extent| {
                acc.saturating_mul(*extent)
            });
        if self.complex_interleaved {
            base.saturating_mul(2)
        } else {
            base
        }
    }
}

#[derive(Clone, Debug)]
pub struct OracleRawInputs {
    pub atm: Vec<i32>,
    pub bas: Vec<i32>,
    pub env: Vec<f64>,
    shls2: Vec<i32>,
    shls3: Vec<i32>,
    shls4: Vec<i32>,
}

impl OracleRawInputs {
    /// Single-atom (Z=1 at origin), 4 single-primitive shell fixture with a
    /// libcint-conformant `env` layout.
    ///
    /// `env[0..PTR_ENV_START(=20)]` is reserved for libcint global parameters and
    /// is left zero EXCEPT the two legitimate grid slots that live inside that
    /// region: `env[NGRIDS(=11)] = 1.0` (one grid point) and
    /// `env[PTR_GRIDS(=12)] = <grid-coord index>` (the env index, `>= 20`, where
    /// the grid coordinate triple begins).
    ///
    /// Critically, no shell coefficient lands on `env[PTR_RANGE_OMEGA(=8)]` (the
    /// range-separation ω). The previous packed layout stored shell-2's coeff
    /// (0.6) there, causing vendored libcint to compute RANGE-SEPARATED (erf,
    /// ω=0.6) 2e integrals while cintx computed full Coulomb — the cint2e
    /// legacy-parity divergence. With ω=0.0 both sides compute full Coulomb.
    ///
    /// Physical basis (unchanged): single atom Z=1 at origin; shells
    /// 0:(l=0, exp=1.0, coeff=1.0), 1:(l=1, exp=0.9, coeff=0.8),
    /// 2:(l=0, exp=0.7, coeff=0.6), 3:(l=1, exp=0.5, coeff=0.4); one grid point
    /// at the origin. shls2/3/4 and per-shell sizes are preserved exactly.
    ///
    /// Mirrors the conformant `build_h2o_sto3g()` pattern: reserve
    /// `PTR_ENV_START` zero slots, then append user data with growing `env.len()`
    /// pointers, using the named `PTR_*` / ATM-slot constants.
    pub fn sample() -> Self {
        // 1. Reserve env[0..PTR_ENV_START] for libcint global params (zeros).
        let mut env = vec![0.0_f64; PTR_ENV_START];

        // 2. One grid point lives in the reserved region at env[NGRIDS]; the
        //    grid-coord index env[PTR_GRIDS] is filled in step 5 once known.
        env[NGRIDS] = 1.0;

        // 2b. F12/STG/YP correlation-factor exponent on its designated libcint
        //     reserved slot. Required so the shared-inputs.env profile parity can
        //     evaluate the F12-family fixtures (int2e_stg_*/int2e_yp_*): the cintx
        //     runtime validator fail-closes F12 plans when env[PTR_F12_ZETA] is
        //     0.0. 1.2 is the documented typical zeta (mirrors build_h2o_sto3g_f12).
        //     Non-F12 integrals ignore env[9], so base/with-4c1e and all Coulomb
        //     results are unchanged; env[PTR_RANGE_OMEGA]=env[8] stays 0.0.
        env[PTR_F12_ZETA] = 1.2;

        // 3. Atom coordinate (0,0,0) at >= PTR_ENV_START.
        let coord_ptr = env.len() as i32;
        env.extend_from_slice(&[0.0, 0.0, 0.0]);

        // 4. Per-shell (exp, coeff) pairs, capturing each pointer before push.
        let shell0_exp_ptr = env.len() as i32;
        env.push(1.0);
        let shell0_coeff_ptr = env.len() as i32;
        env.push(1.0);

        let shell1_exp_ptr = env.len() as i32;
        env.push(0.9);
        let shell1_coeff_ptr = env.len() as i32;
        env.push(0.8);

        let shell2_exp_ptr = env.len() as i32;
        env.push(0.7);
        let shell2_coeff_ptr = env.len() as i32;
        env.push(0.6);

        let shell3_exp_ptr = env.len() as i32;
        env.push(0.5);
        let shell3_coeff_ptr = env.len() as i32;
        env.push(0.4);

        // 5. Grid coordinates (one point at origin); record its index in env[PTR_GRIDS].
        let grid_ptr = env.len() as i32;
        env.extend_from_slice(&[0.0, 0.0, 0.0]);
        env[PTR_GRIDS] = grid_ptr as f64;

        // atm: single point-charge atom Z=1 at origin. PTR_ZETA points at the
        // reserved zero slot 0 (reads 0.0, fine for a point nucleus).
        let mut atm = vec![0_i32; ATM_SLOTS];
        atm[CHARGE_OF] = 1;
        atm[PTR_COORD] = coord_ptr;
        atm[NUC_MOD_OF] = POINT_NUC;
        atm[PTR_ZETA] = 0;

        // bas: per shell [atom=0, l, nprim=1, nctr=1, kappa=0, ptr_exp, ptr_coeff, 0].
        let mut bas = vec![0_i32; 4 * BAS_SLOTS];

        bas[0 * BAS_SLOTS + ATOM_OF] = 0;
        bas[0 * BAS_SLOTS + ANG_OF] = 0;
        bas[0 * BAS_SLOTS + NPRIM_OF] = 1;
        bas[0 * BAS_SLOTS + NCTR_OF] = 1;
        bas[0 * BAS_SLOTS + PTR_EXP] = shell0_exp_ptr;
        bas[0 * BAS_SLOTS + PTR_COEFF] = shell0_coeff_ptr;

        bas[1 * BAS_SLOTS + ATOM_OF] = 0;
        bas[1 * BAS_SLOTS + ANG_OF] = 1;
        bas[1 * BAS_SLOTS + NPRIM_OF] = 1;
        bas[1 * BAS_SLOTS + NCTR_OF] = 1;
        bas[1 * BAS_SLOTS + PTR_EXP] = shell1_exp_ptr;
        bas[1 * BAS_SLOTS + PTR_COEFF] = shell1_coeff_ptr;

        bas[2 * BAS_SLOTS + ATOM_OF] = 0;
        bas[2 * BAS_SLOTS + ANG_OF] = 0;
        bas[2 * BAS_SLOTS + NPRIM_OF] = 1;
        bas[2 * BAS_SLOTS + NCTR_OF] = 1;
        bas[2 * BAS_SLOTS + PTR_EXP] = shell2_exp_ptr;
        bas[2 * BAS_SLOTS + PTR_COEFF] = shell2_coeff_ptr;

        bas[3 * BAS_SLOTS + ATOM_OF] = 0;
        bas[3 * BAS_SLOTS + ANG_OF] = 1;
        bas[3 * BAS_SLOTS + NPRIM_OF] = 1;
        bas[3 * BAS_SLOTS + NCTR_OF] = 1;
        bas[3 * BAS_SLOTS + PTR_EXP] = shell3_exp_ptr;
        bas[3 * BAS_SLOTS + PTR_COEFF] = shell3_coeff_ptr;

        Self {
            atm,
            bas,
            env,
            shls2: vec![0, 1],
            shls3: vec![0, 1, 2],
            shls4: vec![0, 1, 2, 3],
        }
    }

    pub fn shells_for_arity(&self, arity: usize) -> &[i32] {
        match arity {
            2 => &self.shls2,
            3 => &self.shls3,
            4 => &self.shls4,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactWriteResult {
    pub required_path: &'static str,
    pub actual_path: PathBuf,
    pub used_required_path: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct LockSymbolMetadata {
    pub(crate) profiles: BTreeSet<String>,
    pub(crate) stability: String,
    pub(crate) oracle_covered: bool,
}

pub fn write_pretty_json_artifact(
    required_path: &'static str,
    fallback_name: &str,
    value: &Value,
) -> Result<ArtifactWriteResult> {
    let payload = serde_json::to_vec_pretty(value).context("serialize artifact json")?;
    let required = Path::new(required_path);
    match try_write_payload(required, &payload) {
        Ok(()) => Ok(ArtifactWriteResult {
            required_path,
            actual_path: required.to_path_buf(),
            used_required_path: true,
            fallback_reason: None,
        }),
        Err(error) => {
            let fallback_dir = std::env::var(FALLBACK_ARTIFACT_DIR_ENV)
                .unwrap_or_else(|_| FALLBACK_ARTIFACT_DIR_DEFAULT.to_owned());
            let fallback = Path::new(&fallback_dir).join(fallback_name);
            try_write_payload(&fallback, &payload).with_context(|| {
                format!(
                    "failed to write fallback artifact `{}` after required-path failure",
                    fallback.display()
                )
            })?;
            Ok(ArtifactWriteResult {
                required_path,
                actual_path: fallback,
                used_required_path: false,
                fallback_reason: Some(error.to_string()),
            })
        }
    }
}

fn try_write_payload(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create artifact parent directory `{}`", parent.display()))?;
    }
    fs::write(path, payload).with_context(|| format!("write artifact `{}`", path.display()))?;
    Ok(())
}

fn representation_from_entry(entry: &ManifestEntry) -> Option<Representation> {
    match (
        entry.representation.cart,
        entry.representation.spheric,
        entry.representation.spinor,
    ) {
        (true, false, false) => Some(Representation::Cart),
        (false, true, false) => Some(Representation::Spheric),
        (false, false, true) => Some(Representation::Spinor),
        _ => None,
    }
}

fn representation_name(representation: Representation) -> &'static str {
    match representation {
        Representation::Cart => "cart",
        Representation::Spheric => "spheric",
        Representation::Spinor => "spinor",
    }
}

fn parse_component_count(component_rank: &str) -> Result<usize> {
    let trimmed = component_rank.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("scalar") {
        return Ok(1);
    }

    let mut count = 1usize;
    let mut found = false;
    for segment in trimmed.split(|ch: char| !ch.is_ascii_digit()) {
        if segment.is_empty() {
            continue;
        }
        let value = segment.parse::<usize>().with_context(|| {
            format!("parse component segment `{segment}` from `{component_rank}`")
        })?;
        count = count
            .checked_mul(value)
            .ok_or_else(|| anyhow!("component rank overflow for `{component_rank}`"))?;
        found = true;
    }
    if !found {
        return Ok(1);
    }
    Ok(count)
}

fn oracle_component_count(entry: &ManifestEntry) -> Result<usize> {
    // Grids derivatives are emitted with source-manifest component_rank values that
    // do not directly equal runtime component multiplicity. Oracle fixtures must
    // use the runtime output contract (ncomp) to size raw buffers correctly.
    let overridden = match entry.symbol_name {
        "int1e_grids_sph" => Some(1),
        "int1e_grids_ip_sph" => Some(3),
        "int1e_grids_ipvip_sph" | "int1e_grids_ipip_sph" => Some(9),
        "int1e_grids_spvsp_sph" => Some(4),
        _ => None,
    };
    if let Some(value) = overridden {
        return Ok(value);
    }
    parse_component_count(entry.component_rank)
}

fn ao_count_for_rep(shell: i32, representation: Representation, bas: &[i32]) -> Result<usize> {
    match representation {
        Representation::Cart => {
            CINTcgto_cart(shell, bas).with_context(|| format!("cart ao count for shell {shell}"))
        }
        Representation::Spheric => CINTcgto_spheric(shell, bas)
            .with_context(|| format!("spheric ao count for shell {shell}")),
        Representation::Spinor => CINTcgto_spinor(shell, bas)
            .with_context(|| format!("spinor ao count for shell {shell}")),
    }
}

fn dims_for_arity(
    inputs: &OracleRawInputs,
    representation: Representation,
    arity: usize,
) -> Result<Vec<usize>> {
    let shells = inputs.shells_for_arity(arity);
    shells
        .iter()
        .copied()
        .enumerate()
        .map(|(axis, shell)| {
            // Arity-3 SPINOR families size the auxiliary-k axis (the tail shell,
            // axis == arity - 1) SPHERICALLY as nsph(lk) = (2lk+1)*nctr_k, NOT spinor.
            // Source-verified: libcint CINT3c2e_spinor_drv is_ssc=0 branch
            // (cint3c2e.c:631-636) sets counts[2] = (k_l*2+1)*x_ctr[2]; only bra i
            // and ket j use CINTcgto_spinor (4l+2). See 27-SPIKE-FINDINGS CORRECTION
            // NOTICE — the earlier spinor-sized aux-k (the disproven 720) was a
            // compat-dims over-sizing bug in this function.
            if representation == Representation::Spinor && arity == 3 && axis == arity - 1 {
                CINTcgto_spheric(shell, &inputs.bas).with_context(|| {
                    format!("spherical aux-k ao count for arity-3 spinor shell {shell}")
                })
            } else {
                ao_count_for_rep(shell, representation, &inputs.bas)
            }
        })
        .collect()
}

/// Derive oracle-eligible families from the manifest lock.
/// Any entry with stability "stable" or "optional" is oracle-eligible.
/// Replaces the hardcoded PHASE4_ORACLE_FAMILIES constant.
pub fn manifest_oracle_families() -> BTreeSet<String> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON)
        .expect("compiled manifest lock JSON parse");
    root["entries"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|e| {
            let stab = e.get("stability").and_then(Value::as_str).unwrap_or("");
            if matches!(stab, "stable" | "optional") {
                e.get("id")
                    .and_then(|id| id.get("family"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Check if a family is oracle-eligible based on manifest or unstable-source prefix.
pub fn is_oracle_eligible_family(family: &str) -> bool {
    manifest_oracle_families().contains(family) || family.starts_with("unstable::source::")
}

fn is_phase4_oracle_family(family: &str) -> bool {
    is_oracle_eligible_family(family)
}

fn stability_is_included(stability: &str, include_unstable_source: bool) -> bool {
    match stability {
        "stable" | "optional" => true,
        "unstable_source" => include_unstable_source,
        _ => false,
    }
}

fn ensure_profile_approved(profile: &str) -> Result<()> {
    if ORACLE_COMPARE_APPROVED_PROFILES.contains(&profile) {
        return Ok(());
    }
    bail!(
        "unsupported profile `{profile}`; expected one of {:?}",
        ORACLE_COMPARE_APPROVED_PROFILES
    )
}

pub(crate) fn manifest_lock_symbol_metadata() -> Result<BTreeMap<String, LockSymbolMetadata>> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON)
        .context("parse compiled manifest lock")?;
    let entries = root
        .get("entries")
        .and_then(Value::as_array)
        .context("compiled manifest lock missing `entries` array")?;

    let mut symbols = BTreeMap::new();
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(Value::as_object)
            .context("compiled manifest entry missing `id`")?;
        let family = id.get("family").and_then(Value::as_str).unwrap_or_default();
        if !is_phase4_oracle_family(family) {
            continue;
        }
        // ECP and other dedicated-harness families are oracle-covered but verified
        // outside the generic raw-eval matrix; exclude them from the expected-symbol
        // set so the matrix completeness check stays consistent (see
        // DEDICATED_ORACLE_FAMILIES). Matched on `canonical_family`, since ECP's
        // `id.family` is `"1e"`.
        let canonical_family = entry
            .get("canonical_family")
            .and_then(Value::as_str)
            .unwrap_or(family);
        if is_dedicated_oracle_family(canonical_family) {
            continue;
        }
        let Some(symbol) = id.get("symbol").and_then(Value::as_str) else {
            continue;
        };
        let Some(_representation) = id.get("representation").and_then(Value::as_str) else {
            continue;
        };

        let profiles = entry
            .get("profiles")
            .and_then(Value::as_array)
            .map(|profiles| {
                profiles
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if profiles.is_empty() {
            continue;
        }
        let stability = entry
            .get("stability")
            .and_then(Value::as_str)
            .unwrap_or("stable")
            .to_owned();
        let oracle_covered = entry
            .get("oracle_covered")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        symbols.insert(
            symbol.to_owned(),
            LockSymbolMetadata {
                profiles,
                stability,
                oracle_covered,
            },
        );
    }

    Ok(symbols)
}

fn phase4_operator_entries(
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<&'static ManifestEntry>> {
    ensure_profile_approved(profile)?;
    let metadata = manifest_lock_symbol_metadata()?;

    let mut entries = Vec::new();
    for entry in Resolver::manifest() {
        if !matches!(
            entry.helper_kind,
            HelperKind::Operator | HelperKind::SourceOnly
        ) {
            continue;
        }
        if !is_phase4_oracle_family(entry.family_name) {
            continue;
        }
        // ECP and other dedicated-harness families are oracle-covered but verified
        // outside the generic raw-eval matrix (see DEDICATED_ORACLE_FAMILIES); skip
        // them here so they never enter the representation matrix. This must precede
        // the metadata lookup below, since `manifest_lock_symbol_metadata` already
        // omits these symbols.
        if is_dedicated_oracle_family(entry.canonical_family) {
            continue;
        }
        let Some(lock_entry) = metadata.get(entry.symbol_name) else {
            bail!(
                "manifest lock metadata missing for oracle symbol `{}`",
                entry.symbol_name
            );
        };
        let profile_matches = lock_entry.profiles.contains(profile)
            || (include_unstable_source && lock_entry.stability == "unstable_source");
        if !profile_matches {
            continue;
        }
        if !stability_is_included(&lock_entry.stability, include_unstable_source) {
            continue;
        }
        entries.push(entry);
    }
    entries.sort_by_key(|entry| entry.symbol_name);
    Ok(entries)
}

fn phase2_operator_entries() -> Result<Vec<&'static ManifestEntry>> {
    let entries = phase4_operator_entries(BASE_PROFILE, false)?;
    Ok(entries
        .into_iter()
        .filter(|entry| PHASE2_FAMILIES.contains(&entry.family_name))
        .collect())
}

pub fn phase2_manifest_symbols() -> BTreeSet<String> {
    phase2_operator_entries()
        .expect("phase2 manifest symbols")
        .into_iter()
        .map(|entry| entry.symbol_name.to_owned())
        .collect()
}

pub fn manifest_lock_symbols_for_profile(
    profile: &str,
    include_unstable_source: bool,
) -> Result<BTreeSet<String>> {
    ensure_profile_approved(profile)?;
    let metadata = manifest_lock_symbol_metadata()?;
    Ok(metadata
        .into_iter()
        .filter(|(_, value)| {
            value.profiles.contains(profile)
                || (include_unstable_source && value.stability == "unstable_source")
        })
        .filter(|(_, value)| stability_is_included(&value.stability, include_unstable_source))
        .map(|(symbol, _)| symbol)
        .collect())
}

pub fn manifest_lock_symbols() -> Result<BTreeSet<String>> {
    Ok(manifest_lock_symbols_for_profile(BASE_PROFILE, false)?
        .into_iter()
        .filter(|symbol| {
            Resolver::manifest().iter().any(|entry| {
                entry.symbol_name == symbol
                    && matches!(entry.helper_kind, HelperKind::Operator)
                    && PHASE2_FAMILIES.contains(&entry.family_name)
            })
        })
        .collect())
}

pub fn build_profile_representation_matrix(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<OracleFixture>> {
    let mut fixtures = Vec::new();
    for entry in phase4_operator_entries(profile, include_unstable_source)? {
        let Some(representation) = representation_from_entry(entry) else {
            continue;
        };
        let dims = dims_for_arity(inputs, representation, usize::from(entry.arity))
            .with_context(|| format!("derive dims for `{}`", entry.symbol_name))?;
        fixtures.push(OracleFixture {
            family: entry.family_name.to_owned(),
            symbol: entry.symbol_name.to_owned(),
            representation: representation_name(representation).to_owned(),
            arity: usize::from(entry.arity),
            dims,
            component_count: oracle_component_count(entry)
                .with_context(|| format!("component_count for `{}`", entry.symbol_name))?,
            complex_interleaved: matches!(representation, Representation::Spinor)
                || entry.complex_output,
        });
    }
    fixtures.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    Ok(fixtures)
}

pub fn build_required_profile_matrices(
    inputs: &OracleRawInputs,
) -> Result<Vec<ProfileRepresentationMatrix>> {
    PHASE4_APPROVED_PROFILES
        .iter()
        .copied()
        .map(|profile| {
            let fixtures = build_profile_representation_matrix(inputs, profile, false)?;
            Ok(ProfileRepresentationMatrix {
                profile: profile.to_owned(),
                fixtures,
            })
        })
        .collect()
}

pub fn build_phase2_representation_matrix(inputs: &OracleRawInputs) -> Result<Vec<OracleFixture>> {
    Ok(
        build_profile_representation_matrix(inputs, BASE_PROFILE, false)?
            .into_iter()
            .filter(|fixture| PHASE2_FAMILIES.contains(&fixture.family.as_str()))
            .collect(),
    )
}

pub fn write_profile_representation_matrix_artifact(
    profile: &str,
    include_unstable_source: bool,
    matrix: &[OracleFixture],
) -> Result<ArtifactWriteResult> {
    let artifact = build_matrix_artifact_json(profile, include_unstable_source, matrix)?;
    write_pretty_json_artifact(
        REQUIRED_MATRIX_ARTIFACT,
        MATRIX_ARTIFACT_FALLBACK_NAME,
        &artifact,
    )
}

fn build_matrix_artifact_json(
    profile: &str,
    include_unstable_source: bool,
    matrix: &[OracleFixture],
) -> Result<Value> {
    ensure_profile_approved(profile)?;

    let fixture_symbols: BTreeSet<&str> = matrix
        .iter()
        .map(|fixture| fixture.symbol.as_str())
        .collect();
    let expected_symbols = manifest_lock_symbols_for_profile(profile, include_unstable_source)?;
    let missing_symbols: Vec<String> = expected_symbols
        .iter()
        .filter(|symbol| !fixture_symbols.contains(symbol.as_str()))
        .cloned()
        .collect();
    if !missing_symbols.is_empty() {
        bail!(
            "fixture matrix for profile `{profile}` is missing {} symbols from compiled manifest lock",
            missing_symbols.len()
        );
    }

    let matrix_families: BTreeSet<&str> = matrix
        .iter()
        .map(|fixture| fixture.family.as_str())
        .collect();
    if matrix
        .iter()
        .any(|fixture| fixture.family.starts_with("unstable::source::"))
        && !include_unstable_source
    {
        bail!("fixture matrix unexpectedly contains unstable_source rows while disabled");
    }

    let fixtures_json: Vec<Value> = matrix
        .iter()
        .map(|fixture| {
            json!({
                "family": fixture.family,
                "symbol": fixture.symbol,
                "representation": fixture.representation,
                "arity": fixture.arity,
                "dims": fixture.dims,
                "component_count": fixture.component_count,
                "complex_interleaved": fixture.complex_interleaved,
                "required_elements": fixture.required_elements(),
            })
        })
        .collect();

    Ok(json!({
        "profile": profile,
        "include_unstable_source": include_unstable_source,
        "representation_matrix": fixtures_json,
        "fixture_count": matrix.len(),
        "required_path": REQUIRED_MATRIX_ARTIFACT,
        "compiled_manifest": "crates/cintx-ops/generated/compiled_manifest.lock.json",
        "approved_profiles": PHASE4_APPROVED_PROFILES,
        "oracle_families": manifest_oracle_families().into_iter().collect::<Vec<_>>(),
        "matrix_families": matrix_families,
    }))
}

pub fn write_representation_matrix_artifact(
    matrix: &[OracleFixture],
) -> Result<ArtifactWriteResult> {
    write_profile_representation_matrix_artifact(BASE_PROFILE, false, matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libcint range-separation ω slot inside the reserved env[0..PTR_ENV_START]
    /// region. Not exported as a named constant by cintx-compat (it carries no
    /// cintx semantics), so it is declared locally here to assert the specific
    /// collision being fixed: a shell coeff must never land on this slot.
    const PTR_RANGE_OMEGA: usize = 8;

    #[test]
    fn sample_env_reserves_libcint_global_slots() {
        let inputs = OracleRawInputs::sample();

        // The specific collision being fixed: env[8]=PTR_RANGE_OMEGA must be 0.0
        // so vendored libcint computes FULL Coulomb (not range-separated) 2e ints.
        assert_eq!(
            inputs.env[PTR_RANGE_OMEGA], 0.0,
            "env[PTR_RANGE_OMEGA] must be 0.0 (full-Coulomb), not a shell coeff"
        );

        // env[PTR_GRIDS] holds the grid-coord index, which must be >= PTR_ENV_START.
        let grid_coord_index = inputs.env[PTR_GRIDS];
        assert!(
            grid_coord_index >= PTR_ENV_START as f64,
            "env[PTR_GRIDS]={grid_coord_index} must point at user data (>= {PTR_ENV_START})"
        );

        // Every reserved slot env[0..PTR_ENV_START] is 0.0 EXCEPT the two
        // legitimate grid slots — env[NGRIDS] (==1.0) and env[PTR_GRIDS] (the
        // index) — and env[PTR_F12_ZETA] (==1.2, the F12/STG/YP correlation-factor
        // exponent). env[PTR_RANGE_OMEGA]=env[8] must STILL be 0.0 (asserted above).
        for slot in 0..PTR_ENV_START {
            if slot == NGRIDS {
                assert_eq!(
                    inputs.env[slot], 1.0,
                    "env[NGRIDS] must be 1.0 (one grid point)"
                );
            } else if slot == PTR_GRIDS {
                assert_eq!(
                    inputs.env[slot], grid_coord_index,
                    "env[PTR_GRIDS] must hold the grid-coord index"
                );
            } else if slot == PTR_F12_ZETA {
                assert_eq!(
                    inputs.env[slot], 1.2,
                    "env[PTR_F12_ZETA] must be 1.2 (F12 correlation-factor exponent)"
                );
            } else {
                assert_eq!(
                    inputs.env[slot], 0.0,
                    "reserved env[{slot}] must be 0.0 (libcint global param)"
                );
            }
        }

        // shls2/3/4 unchanged.
        assert_eq!(inputs.shells_for_arity(2), &[0, 1]);
        assert_eq!(inputs.shells_for_arity(3), &[0, 1, 2]);
        assert_eq!(inputs.shells_for_arity(4), &[0, 1, 2, 3]);

        // Physical basis shape preserved: 4 shells, angular momenta [0,1,0,1].
        assert_eq!(inputs.bas.len(), 4 * BAS_SLOTS, "must be 4 shells");
        let ang: Vec<i32> = (0..4).map(|s| inputs.bas[s * BAS_SLOTS + ANG_OF]).collect();
        assert_eq!(ang, vec![0, 1, 0, 1], "angular momenta must be s,p,s,p");
    }

    #[test]
    fn representation_matrix_matches_manifest_fixtures() {
        let inputs = OracleRawInputs::sample();
        let matrix = build_phase2_representation_matrix(&inputs).expect("matrix");
        let actual: BTreeSet<String> = matrix
            .iter()
            .map(|fixture| fixture.symbol.clone())
            .collect();

        let expected = phase2_manifest_symbols();
        assert_eq!(actual, expected);

        let lock = manifest_lock_symbols().expect("lock symbols");
        assert_eq!(actual, lock);
    }

    #[test]
    fn required_profile_matrices_match_manifest_profiles() {
        let inputs = OracleRawInputs::sample();
        let matrices = build_required_profile_matrices(&inputs).expect("required matrices");
        let actual_profiles: Vec<String> = matrices
            .iter()
            .map(|matrix| matrix.profile.clone())
            .collect();
        let expected_profiles: Vec<String> = PHASE4_APPROVED_PROFILES
            .iter()
            .map(|profile| (*profile).to_owned())
            .collect();
        assert_eq!(actual_profiles, expected_profiles);

        for matrix in matrices {
            let symbols: BTreeSet<String> = matrix
                .fixtures
                .iter()
                .map(|fixture| fixture.symbol.clone())
                .collect();
            let expected = manifest_lock_symbols_for_profile(&matrix.profile, false)
                .expect("profile lock symbols");
            assert_eq!(symbols, expected, "profile {} mismatch", matrix.profile);
        }
    }

    #[test]
    fn unstable_source_fixtures_require_opt_in() {
        let inputs = OracleRawInputs::sample();
        let stable_only =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, false).expect("stable");
        assert!(
            stable_only
                .iter()
                .all(|fixture| !fixture.family.starts_with("unstable::source::")),
            "stable run should exclude unstable_source fixtures"
        );

        let with_unstable =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, true).expect("unstable");
        assert!(
            with_unstable
                .iter()
                .any(|fixture| fixture.family.starts_with("unstable::source::")),
            "explicit unstable_source mode should include source-only fixtures"
        );
    }

    #[test]
    fn unstable_source_profile_is_accepted_when_enabled() {
        let inputs = OracleRawInputs::sample();
        let matrix = build_profile_representation_matrix(&inputs, "unstable-source", true)
            .expect("unstable-source profile should be accepted with opt-in");
        assert!(
            !matrix.is_empty(),
            "unstable-source profile should include oracle fixtures"
        );
        assert!(
            matrix
                .iter()
                .all(|fixture| fixture.family.starts_with("unstable::source::")),
            "unstable-source profile should only include unstable families"
        );
    }

    #[test]
    fn unstable_source_grids_component_counts_match_runtime_contract() {
        let inputs = OracleRawInputs::sample();
        let matrix = build_profile_representation_matrix(&inputs, "unstable-source", true)
            .expect("unstable-source profile matrix");
        let component_for = |symbol: &str| -> usize {
            matrix
                .iter()
                .find(|fixture| fixture.symbol == symbol)
                .expect("symbol in unstable-source matrix")
                .component_count
        };
        assert_eq!(component_for("int1e_grids_sph"), 1);
        assert_eq!(component_for("int1e_grids_ip_sph"), 3);
        assert_eq!(component_for("int1e_grids_ipvip_sph"), 9);
        assert_eq!(component_for("int1e_grids_spvsp_sph"), 4);
        assert_eq!(component_for("int1e_grids_ipip_sph"), 9);
    }

    #[test]
    fn representation_matrix_artifact_is_written() {
        // Build the matrix and serialize it through the same code path as
        // write_representation_matrix_artifact, but write to an isolated temp
        // file to avoid races with parallel tests that share the fallback dir.
        let inputs = OracleRawInputs::sample();
        let matrix =
            build_profile_representation_matrix(&inputs, BASE_PROFILE, false).expect("matrix");

        let tmp_dir = std::env::temp_dir().join(format!(
            "cintx_matrix_artifact_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = fs::create_dir_all(&tmp_dir);
        let artifact_path = tmp_dir.join(MATRIX_ARTIFACT_FALLBACK_NAME);

        let artifact =
            build_matrix_artifact_json(BASE_PROFILE, false, &matrix).expect("artifact json");
        let payload = serde_json::to_vec_pretty(&artifact).expect("serialize");
        fs::write(&artifact_path, &payload).expect("write artifact");

        assert!(artifact_path.is_file());
        let content = fs::read_to_string(&artifact_path).expect("artifact content");
        assert!(content.contains("representation_matrix"));
        assert!(content.contains(REQUIRED_MATRIX_ARTIFACT));
        assert!(content.contains("\"profile\": \"base\""));

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    /// D-02: build_kappa_spinor_2e_fixture must be a 4-shell, non-square, GT/LT-mix,
    /// nctr>1 quartet. Asserts every hard constraint so a regression that squares the
    /// block or drops the kappa mix / general contraction fails loudly.
    #[test]
    fn kappa_spinor_2e_fixture_meets_d02_constraints() {
        use cintx_compat::raw::KAPPA_OF;
        use cintx_cubecl::transform::c2spinor::spinor_len;

        let (_atm, bas, _env) = build_kappa_spinor_2e_fixture();

        // exactly 4 spinor shells
        let n_shells = bas.len() / BAS_SLOTS;
        assert_eq!(n_shells, 4, "fixture must have exactly 4 spinor shells");

        // collect (l, kappa) per shell and derive spinor dims
        let mut dims = Vec::with_capacity(4);
        let mut has_gt = false; // kappa < 0
        let mut has_lt = false; // kappa > 0
        let mut has_nctr_gt1 = false;
        for s in 0..4 {
            let l = bas[s * BAS_SLOTS + ANG_OF] as u8;
            let kappa = bas[s * BAS_SLOTS + KAPPA_OF];
            let nctr = bas[s * BAS_SLOTS + NCTR_OF];
            assert_ne!(kappa, 0, "shell {s} must have genuine kappa≠0 (GT/LT path)");
            if kappa < 0 {
                has_gt = true;
            }
            if kappa > 0 {
                has_lt = true;
            }
            if nctr > 1 {
                has_nctr_gt1 = true;
            }
            dims.push(spinor_len(l, kappa));
        }

        // GT/LT mix present (exercises BOTH 2l and 2l+2 spinor_len branches)
        assert!(
            has_gt && has_lt,
            "fixture must mix GT (kappa<0) and LT (kappa>0) shells"
        );
        // ≥1 shell with nctr>1 (catches the column/row-major coeff transpose)
        assert!(has_nctr_gt1, "fixture must have ≥1 shell with nctr>1");

        // NON-SQUARE: not all four spinor dims equal (defeats transpose symmetry)
        let all_equal = dims.iter().all(|&d| d == dims[0]);
        assert!(
            !all_equal,
            "fixture spinor dims must NOT all be equal (non-square): {dims:?}"
        );

        // Concrete sizing check: the LT path (2l) and GT path (2l+2) are both exercised.
        // i: p kappa=+1 → 2 (LT);  j: d kappa=−1 → 6 (GT);  k: s kappa=−1 → 2;  l: p kappa=−1 → 4
        assert_eq!(dims, vec![2, 6, 2, 4], "expected spinor dims (2,6,2,4)");
    }

    /// Secondary realism cross-check: the heavy-atom fixture is asserted FINITE only,
    /// NOT the primary gate (RESEARCH §Sampling Rate). Here we assert it is constructible
    /// and well-formed (2 shells, finite env) so it can serve as the realism guard.
    #[test]
    fn heavy_atom_spinor_fixture_is_well_formed() {
        let (_atm, bas, env) = build_heavy_atom_spinor_fixture();
        assert_eq!(
            bas.len() / BAS_SLOTS,
            2,
            "heavy-atom realism fixture is 2 shells"
        );
        assert!(
            env.iter().all(|v| v.is_finite()),
            "heavy-atom env must be finite"
        );
    }
}
