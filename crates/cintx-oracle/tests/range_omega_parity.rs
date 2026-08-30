//! D-PBC-24 — range-separated Coulomb (`env[PTR_RANGE_OMEGA]`) parity sweep.
//!
//! # What this file is
//!
//! Stage 0 of the D-PBC-24 plan: the pre-implementation measurement, and the
//! stage-2 acceptance gate it turns into. It sweeps `env[8]` over
//! `{0, +0.3, +0.8, −0.3, −0.8}` for `int2e`, `int3c2e` and `int2c2e` on shell
//! tuples covering `rys_order ∈ {1,2,3}` (the doubled-root regime, where SR is
//! "full minus long range" with no new quadrature) and `{4,5}` (the
//! `CINTsr_rys_roots` regime, which cintx does not implement and must REFUSE).
//!
//! # Why range separation is not a new integral symbol
//!
//! libcint has no `int2e_sr_*` and PySCF never asks for one:
//! `pyscf/pbc/df/rsjk.py:186` sets `supmol_sr.omega = -self.omega` and calls the
//! *standard* `int2e`. So both sides of every comparison below call the SAME
//! symbol, with the same `atm`/`bas`, and differ only in `env[8]`.
//!
//! # The three assertions
//!
//! 1. **Vendor parity.** cintx == vendored libcint 6.1.3, at every ω, on every
//!    tuple in the doubled-root regime.
//! 2. **`SR(ω) + LR(ω) == full`.** The one check that catches an `erf`/`erfc`
//!    swap, and the identity `pyscf-pbc-df` already gates its `weighted_coulG`
//!    half of at exactly 0.
//! 3. **Fail-closed above `rys_order = 3`.** Short range there needs the
//!    lower-bounded quadrature; cintx must refuse, not substitute the
//!    full-range kernel. A full-range substitute runs, converges, and is
//!    silently a different method.
//!
//! Vendor comparisons require `CINTX_ORACLE_BUILD_VENDOR=1`; without it the
//! identity and fail-closed tests still run against cintx alone.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RANGE_OMEGA, PTR_ZETA, RawApiId, eval_raw,
};

/// The ω sweep from D-PBC-24 stage 0.
const OMEGA_SWEEP: [f64; 5] = [0.0, 0.3, 0.8, -0.3, -0.8];

/// Angular momenta of the eight shells in [`build_spdf_fixture`]: `s p d f` on
/// centre A (indices 0..3) and the same four on centre B (indices 4..7).
const SHELL_L: [i32; 8] = [0, 1, 2, 3, 0, 1, 2, 3];

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// A two-atom fixture carrying `s`, `p`, `d`, `f` on EACH centre.
///
/// Every tuple in [`sweep_tuples`] therefore straddles both centres, so
/// `rr = |R_bra − R_ket|²` is non-zero and the omega-dependent `x = a0 * rr`
/// scaling is genuinely exercised. A same-centre tuple would leave `x = 0`,
/// where the long-range and short-range branches degenerate — and, for
/// `(s,s|d)`, would be identically zero by angular symmetry and would assert
/// nothing at all.
///
/// One primitive and one contraction per shell keeps the reference values easy
/// to reason about while still covering every `rys_order` the sweep needs:
///
/// | integral | shells | `rys_order` | SR regime |
/// |---|---|---|---|
/// | `int2c2e` | (s,s) | 1 | doubled roots |
/// | `int2c2e` | (p,p) | 2 | doubled roots |
/// | `int2c2e` | (d,d) | 3 | doubled roots |
/// | `int2c2e` | (f,f) | 4 | `sr_rys_roots` |
/// | `int3c2e` | (s,s\|d) | 2 | doubled roots |
/// | `int3c2e` | (p,p\|d) | 3 | doubled roots |
/// | `int3c2e` | (d,d\|d) | 4 | `sr_rys_roots` |
/// | `int2e` | (s,s\|s,s) | 1 | doubled roots |
/// | `int2e` | (p,p\|p,p) | 3 | doubled roots |
/// | `int2e` | (d,d\|d,d) | 5 | `sr_rys_roots` |
///
/// The `int3c2e` rows at `rys_order` 2 and 3 are exactly the He-fcc `sto-3g`
/// and diamond `gth-szv` auxiliary classes Phase 14 gates on (D-PBC-24 §3.4).
fn build_spdf_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.3_f64, 0.0, 1.7];

    // Distinct exponents per shell so no two shells coincide.
    let exps = [0.9_f64, 0.7, 1.3, 0.5, 1.1, 0.6, 0.8, 0.4];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let a_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_coord_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut exp_ptr = [0_i32; 8];
    let mut coeff_ptr = [0_i32; 8];
    for (s, &exp) in exps.iter().enumerate() {
        exp_ptr[s] = env.len() as i32;
        env.push(exp);
        coeff_ptr[s] = env.len() as i32;
        env.push(1.0);
    }

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 2;
    atm[PTR_COORD] = a_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 2;
    atm[ATM_SLOTS + PTR_COORD] = b_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 8 * BAS_SLOTS];
    for s in 0..8 {
        bas[s * BAS_SLOTS + ATOM_OF] = if s < 4 { 0 } else { 1 };
        bas[s * BAS_SLOTS + ANG_OF] = SHELL_L[s];
        bas[s * BAS_SLOTS + NPRIM_OF] = 1;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr[s];
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr[s];
    }

    (atm, bas, env)
}

/// A two-centre, one-shell-per-centre fixture parameterised by angular
/// momentum, exponent scale and separation.
///
/// Used by the wide short-range sweep: `lower = sqrt(ω²/(ω² + a0))` is set by
/// the exponents and ω, and `x = a0 · rr` by the separation, so varying these
/// three walks the `CINTsr_rys_roots` dispatch across its whole domain rather
/// than at the handful of points a fixed basis happens to produce.
fn build_two_centre_fixture(l: i32, exp: f64, dist: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, 0.0]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, dist]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let mut exp_ptr = [0_i32; 2];
    let mut coeff_ptr = [0_i32; 2];
    // Distinct exponents so the bra and ket Gaussians are not identical.
    for (s, scale) in [1.0_f64, 1.3].into_iter().enumerate() {
        exp_ptr[s] = env.len() as i32;
        env.push(exp * scale);
        coeff_ptr[s] = env.len() as i32;
        env.push(1.0);
    }

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 2;
    atm[PTR_COORD] = a_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 2;
    atm[ATM_SLOTS + PTR_COORD] = b_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for s in 0..2 {
        bas[s * BAS_SLOTS + ATOM_OF] = s as i32;
        bas[s * BAS_SLOTS + ANG_OF] = l;
        bas[s * BAS_SLOTS + NPRIM_OF] = 1;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptr[s];
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptr[s];
    }

    (atm, bas, env)
}

fn env_with_omega(env: &[f64], omega: f64) -> Vec<f64> {
    let mut env = env.to_vec();
    env[PTR_RANGE_OMEGA] = omega;
    env
}

/// `rys_order = (Σ l)/2 + 1` — the value libcint computes at `g2e.c:74-77`,
/// `g3c2e.c:70` and `g2c2e.c:60`.
fn rys_order(ls: &[i32]) -> usize {
    ls.iter().map(|&l| l as usize).sum::<usize>() / 2 + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx evaluation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn eval_cintx(
    api: RawApiId,
    shls: &[i32],
    out_len: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<Vec<f64>, cintx_core::cintxRsError> {
    let mut out = vec![0.0_f64; out_len];
    unsafe {
        eval_raw(api, Some(&mut out), None, shls, atm, bas, env, None, None)?;
    }
    Ok(out)
}

fn out_len(shls: &[i32], bas: &[i32]) -> usize {
    shls.iter()
        .map(|&s| nsph(bas[s as usize * BAS_SLOTS + ANG_OF]))
        .product()
}

/// The three scalar Coulomb symbols, keyed by tuple arity.
fn api_for_arity(arity: usize) -> RawApiId {
    match arity {
        2 => RawApiId::INT2C2E_SPH,
        3 => RawApiId::INT3C2E_SPH,
        4 => RawApiId::INT2E_SPH,
        other => panic!("no scalar Coulomb symbol of arity {other}"),
    }
}

/// Every tuple in the sweep, as `(label, shell indices)`.
///
/// Shell index `s` has angular momentum `SHELL_L[s]`, so the `rys_order` of a
/// tuple is derived, never hardcoded.
fn sweep_tuples() -> Vec<(&'static str, Vec<i32>)> {
    vec![
        ("int2c2e(s,s)", vec![0, 4]),
        ("int2c2e(p,p)", vec![1, 5]),
        ("int2c2e(d,d)", vec![2, 6]),
        ("int2c2e(f,f)", vec![3, 7]),
        ("int3c2e(s,s|d)", vec![0, 4, 6]),
        ("int3c2e(p,p|d)", vec![1, 5, 6]),
        ("int3c2e(d,d|d)", vec![2, 6, 6]),
        ("int2e(s,s|s,s)", vec![0, 4, 0, 4]),
        ("int2e(p,p|p,p)", vec![1, 5, 1, 5]),
        ("int2e(d,d|d,d)", vec![2, 6, 2, 6]),
    ]
}

fn tuple_ls(shls: &[i32]) -> Vec<i32> {
    shls.iter().map(|&s| SHELL_L[s as usize]).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. SR + LR == full, on every doubled-root tuple
// ─────────────────────────────────────────────────────────────────────────────

/// `erfc(ωr)/r + erf(ωr)/r == 1/r`, evaluated integral by integral.
///
/// This is the check that catches an `erf`/`erfc` swap, a dropped
/// `sqrt(theta)`, or a sign error on the subtracted long-range half — none of
/// which fail loudly. `pyscf-pbc-df` gates the `weighted_coulG` half of the same
/// identity at exactly 0
/// (`tests/rsdf_builder.rs::sr_and_lr_coulg_sum_to_the_full_kernel`); here the
/// tolerance is relative, because the two halves are separately-rounded
/// quadratures rather than closed-form kernels.
#[test]
fn sr_plus_lr_reproduces_the_full_coulomb_integral() {
    let (atm, bas, env) = build_spdf_fixture();
    let natm = 2;
    let nbas = 4;
    let _ = (natm, nbas);

    let mut checked = 0usize;
    for (label, shls) in sweep_tuples() {
        let ls = tuple_ls(&shls);
        let order = rys_order(&ls);
        let api = api_for_arity(shls.len());
        let n = out_len(&shls, &bas);

        let full = eval_cintx(api, &shls, n, &atm, &bas, &env)
            .unwrap_or_else(|e| panic!("{label} full range: {e:?}"));

        for omega in [0.3_f64, 0.8] {
            let lr = eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, omega))
                .unwrap_or_else(|e| panic!("{label} LR omega={omega}: {e:?}"));
            let sr = eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, -omega))
                .unwrap_or_else(|e| panic!("{label} SR omega=-{omega}: {e:?}"));

            // The doubled-root regime computes SR as "full minus long range"
            // over one shared `fac1`, so the identity closes to round-off. The
            // sr_rys_roots regime computes SR from an independently constructed
            // lower-bounded Gauss rule, so it closes to that rule's accuracy —
            // measured at ~3e-12 relative here, and consistent with the
            // integral-level vendor parity below.
            let rel_tol = if order <= 3 { 1e-12 } else { 1e-10 };
            for idx in 0..n {
                let sum = sr[idx] + lr[idx];
                let residual = (sum - full[idx]).abs();
                let tol = rel_tol * full[idx].abs().max(1e-8);
                assert!(
                    residual <= tol,
                    "{label} rys_order={order} omega={omega} idx={idx}: \
                     SR({:.17e}) + LR({:.17e}) = {:.17e} != full({:.17e}), residual={residual:.3e}",
                    sr[idx],
                    lr[idx],
                    sum,
                    full[idx]
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked >= 20,
        "the identity must be checked on EVERY tuple at both omegas — the doubled-root \
         regime and the sr_rys_roots regime alike, got {checked}"
    );
}

/// `omega = 0` written into `env[8]` must be indistinguishable from an unset
/// slot — libcint branches on `omega == 0.` exactly (`g2e.c:4445`).
#[test]
fn an_explicit_zero_omega_is_the_full_coulomb_operator() {
    let (atm, bas, env) = build_spdf_fixture();
    for (label, shls) in sweep_tuples() {
        let api = api_for_arity(shls.len());
        let n = out_len(&shls, &bas);
        let unset = eval_cintx(api, &shls, n, &atm, &bas, &env)
            .unwrap_or_else(|e| panic!("{label} unset: {e:?}"));
        let zero = eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, 0.0))
            .unwrap_or_else(|e| panic!("{label} zero: {e:?}"));
        assert_eq!(
            unset, zero,
            "{label}: env[8] = 0 must be byte-identical to an unset slot"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Fail closed above rys_order 3
// ─────────────────────────────────────────────────────────────────────────────

/// Short range at `rys_order > 3` takes the lower-bounded `CINTsr_rys_roots`
/// quadrature (`rys_roots.c:145`) rather than the doubled-root trick — and it
/// must EVALUATE, not refuse. D-PBC-24 stage 3 landed that family; before it,
/// this same set of tuples was the fail-closed set.
///
/// The one thing that must never happen is a silent fall-through to the
/// full-range kernel: it runs, it converges, and it is a different method. So
/// this asserts both that the values come out and that they are NOT the
/// full-range ones.
#[test]
fn short_range_above_rys_order_three_evaluates_and_is_not_the_full_range_kernel() {
    let (atm, bas, env) = build_spdf_fixture();
    let sr_env = env_with_omega(&env, -0.8);

    let mut checked = 0usize;
    for (label, shls) in sweep_tuples() {
        let order = rys_order(&tuple_ls(&shls));
        if order <= 3 {
            continue;
        }
        let api = api_for_arity(shls.len());
        let n = out_len(&shls, &bas);

        let full = eval_cintx(api, &shls, n, &atm, &bas, &env)
            .unwrap_or_else(|e| panic!("{label} full range: {e:?}"));
        let sr = eval_cintx(api, &shls, n, &atm, &bas, &sr_env).unwrap_or_else(|e| {
            panic!("{label} rys_order={order} short range must evaluate, got {e:?}")
        });

        let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let separation = (0..n).fold(0.0_f64, |m, i| m.max((sr[i] - full[i]).abs()));
        assert!(
            separation > 1e-3 * scale,
            "{label} rys_order={order}: erfc(0.8 r)/r must differ materially from 1/r; \
             a full-range substitute would land here (max |SR - full| = {separation:.3e}, \
             scale = {scale:.3e})"
        );
        checked += 1;
    }
    assert!(
        checked >= 3,
        "the sweep must contain sr_rys_roots-regime tuples, got {checked}"
    );
}

/// Long range has no `rys_order` ceiling — it reuses the ordinary quadrature at
/// a scaled argument (`g2e.c:4493-4512`), so the `sr_rys_roots` refusal above
/// must not leak into it.
#[test]
fn long_range_is_supported_at_every_rys_order() {
    let (atm, bas, env) = build_spdf_fixture();
    let lr_env = env_with_omega(&env, 0.8);
    for (label, shls) in sweep_tuples() {
        let api = api_for_arity(shls.len());
        let n = out_len(&shls, &bas);
        let lr = eval_cintx(api, &shls, n, &atm, &bas, &lr_env)
            .unwrap_or_else(|e| panic!("{label} long range must be supported: {e:?}"));
        assert!(
            lr.iter().any(|v| v.abs() > 1e-18),
            "{label}: long-range output is all zeros"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2c. The derivative rows (D-PBC-24 P2-1)
// ─────────────────────────────────────────────────────────────────────────────

/// One row of the derivative sweep.
struct DerivRow {
    label: &'static str,
    api: RawApiId,
    /// Indices into [`SHELL_L`].
    shls: Vec<i32>,
    /// Component count (3 for a gradient, 9 for a Hessian).
    ncomp: usize,
    /// `(canonical_family, operator_name)` — the key
    /// `cintx_runtime::range_omega::derivative_headroom` is looked up by.
    key: (&'static str, &'static str),
    /// `rys_order` AFTER the `ng[..INC]` raises, hand-computed from
    /// [`SHELL_L`]. Cross-checked against `derivative_headroom` below, so a
    /// table that drifts from the launchers' own `build_2e_shape(li + i_inc, ..)`
    /// calls fails here rather than returning a wrong number.
    rys_order: usize,
}

fn row(
    label: &'static str,
    api: RawApiId,
    shls: &[i32],
    ncomp: usize,
    key: (&'static str, &'static str),
    rys_order: usize,
) -> DerivRow {
    DerivRow {
        label,
        api,
        shls: shls.to_vec(),
        ncomp,
        key,
        rys_order,
    }
}

/// Every derivative row admitted by `supports_range_omega`.
///
/// `int2e_ipip1` on `(p,p|p,p)` raises `li` by 2, so `rys_order = 4` and short
/// range takes the lower-bounded `sr_rys_roots` arm; `int2e_ip1` on the same
/// quartet is `rys_order = 3`, where short range DOUBLES to `nroots = 6` —
/// above `BASE_DEVICE_NROOTS`, which is why these rows route to the host Rys
/// engine and why the raise has to reach the WORKSPACE query and not only the
/// kernel. Both regimes are covered deliberately.
fn derivative_rows() -> Vec<DerivRow> {
    vec![
        // 2e — tuple (i, j, k, l).
        row(
            "int2e_ip1(s,s|s,s)",
            RawApiId::INT2E_IP1_SPH,
            &[0, 4, 0, 4],
            3,
            ("2e", "ip1"),
            1,
        ),
        row(
            "int2e_ip1(p,p|p,p)",
            RawApiId::INT2E_IP1_SPH,
            &[1, 5, 1, 5],
            3,
            ("2e", "ip1"),
            3,
        ),
        row(
            "int2e_ip2(p,p|p,p)",
            RawApiId::INT2E_IP2_SPH,
            &[1, 5, 1, 5],
            3,
            ("2e", "ip2"),
            3,
        ),
        row(
            "int2e_ipip1(s,s|s,s)",
            RawApiId::INT2E_IPIP1_SPH,
            &[0, 4, 0, 4],
            9,
            ("2e", "ipip1"),
            2,
        ),
        row(
            "int2e_ipip1(p,p|p,p)",
            RawApiId::INT2E_IPIP1_SPH,
            &[1, 5, 1, 5],
            9,
            ("2e", "ipip1"),
            4,
        ),
        row(
            "int2e_ipvip1(p,p|p,p)",
            RawApiId::INT2E_IPVIP1_SPH,
            &[1, 5, 1, 5],
            9,
            ("2e", "ipvip1"),
            4,
        ),
        row(
            "int2e_ip1ip2(p,p|p,p)",
            RawApiId::INT2E_IP1IP2_SPH,
            &[1, 5, 1, 5],
            9,
            ("2e", "ip1ip2"),
            4,
        ),
        // 3c2e — tuple (i, j, aux). ip1/ip2 are the rows a range-separated
        // `aux_e2` gradient needs, and the ones that had NO host arm before P2-1.
        row(
            "int3c2e_ip1(s,s|d)",
            RawApiId::INT3C2E_IP1_SPH,
            &[0, 4, 6],
            3,
            ("3c2e", "ip1"),
            2,
        ),
        row(
            "int3c2e_ip1(p,p|d)",
            RawApiId::INT3C2E_IP1_SPH,
            &[1, 5, 6],
            3,
            ("3c2e", "ip1"),
            3,
        ),
        row(
            "int3c2e_ip2(p,p|d)",
            RawApiId::INT3C2E_IP2_SPH,
            &[1, 5, 6],
            3,
            ("3c2e", "ip2"),
            3,
        ),
        row(
            "int3c2e_ipip1(s,s|d)",
            RawApiId::INT3C2E_IPIP1_SPH,
            &[0, 4, 6],
            9,
            ("3c2e", "ipip1"),
            3,
        ),
        row(
            "int3c2e_ipip2(s,s|d)",
            RawApiId::INT3C2E_IPIP2_SPH,
            &[0, 4, 6],
            9,
            ("3c2e", "ipip2"),
            3,
        ),
        row(
            "int3c2e_ipvip1(p,p|d)",
            RawApiId::INT3C2E_IPVIP1_SPH,
            &[1, 5, 6],
            9,
            ("3c2e", "ipvip1"),
            4,
        ),
        row(
            "int3c2e_ip1ip2(p,p|d)",
            RawApiId::INT3C2E_IP1IP2_SPH,
            &[1, 5, 6],
            9,
            ("3c2e", "ip1ip2"),
            4,
        ),
        // 2c2e — tuple (i, k).
        row(
            "int2c2e_ip1(p,p)",
            RawApiId::INT2C2E_IP1_SPH,
            &[1, 5],
            3,
            ("2c2e", "ip1"),
            2,
        ),
        row(
            "int2c2e_ip2(p,p)",
            RawApiId::INT2C2E_IP2_SPH,
            &[1, 5],
            3,
            ("2c2e", "ip2"),
            2,
        ),
        row(
            "int2c2e_ipip1(p,p)",
            RawApiId::INT2C2E_IPIP1_SPH,
            &[1, 5],
            9,
            ("2c2e", "ipip1"),
            3,
        ),
        row(
            "int2c2e_ip1ip2(p,p)",
            RawApiId::INT2C2E_IP1IP2_SPH,
            &[1, 5],
            9,
            ("2c2e", "ip1ip2"),
            3,
        ),
    ]
}

/// The headroom table matches the hand-computed `rys_order` of every row.
///
/// This is the check that says the planner and the launchers agree: the planner
/// sizes the workspace from `derivative_headroom`, and each launcher builds its
/// G tensor from its own `build_2e_shape(li + i_inc, ..)` literal. If the two
/// ever disagree the kernel writes past the queried root count, which is a wrong
/// number rather than a crash — so it is pinned here, cheaply, before the
/// numerics run.
#[test]
fn the_headroom_table_reproduces_each_rows_rys_order() {
    use cintx_runtime::range_omega::{derivative_headroom, rys_order_with_headroom};

    for r in derivative_rows() {
        let headroom = derivative_headroom(r.key.0, r.key.1)
            .unwrap_or_else(|| panic!("{}: {:?} is not in the headroom table", r.label, r.key));
        assert_eq!(
            headroom.len(),
            r.shls.len(),
            "{}: the headroom entry must have one raise per tuple position",
            r.label
        );
        let got = rys_order_with_headroom(
            r.shls.iter().map(|&s| SHELL_L[s as usize] as usize),
            headroom,
        );
        assert_eq!(
            got, r.rys_order,
            "{}: headroom table gives rys_order {got}, hand count says {}",
            r.label, r.rys_order
        );
    }
}

/// The derivative rows honour ω, and close `SR + LR == full` on every
/// component.
///
/// They refused before P2-1, and correctly so: their `rys_order` is
/// `(Σ l + Σ ng[..INC])/2 + 1`, not `(Σ l)/2 + 1`, so a workspace sized from the
/// unraised sum would have been short of the roots the kernel then writes.
/// `derivative_headroom` is the table that closed that, and this sweep is what
/// says the table matches the launchers' own `build_2e_shape(li + i_inc, ..)`
/// calls — a mismatch shows up here as a wrong number, not as a crash.
#[test]
fn derivative_rows_honour_a_set_omega_and_close_sr_plus_lr() {
    let (atm, bas, env) = build_spdf_fixture();

    let mut checked = 0usize;
    for DerivRow {
        label,
        api,
        shls,
        ncomp,
        rys_order,
        ..
    } in derivative_rows()
    {
        let n = out_len(&shls, &bas) * ncomp;

        let full = eval_cintx(api, &shls, n, &atm, &bas, &env)
            .unwrap_or_else(|e| panic!("{label} full range: {e:?}"));
        let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            scale > 1e-12,
            "{label}: the full-range derivative block is all zeros, so nothing below \
             asserts anything"
        );

        for omega in [0.3_f64, 0.8] {
            let lr = eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, omega))
                .unwrap_or_else(|e| panic!("{label} LR omega={omega}: {e:?}"));
            let sr = eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, -omega))
                .unwrap_or_else(|e| panic!("{label} SR omega=-{omega}: {e:?}"));

            let moved = (0..n).fold(0.0_f64, |m, i| m.max((sr[i] - full[i]).abs()));
            assert!(
                moved > 1e-3 * scale,
                "{label} omega=-{omega}: the derivative block is (almost) the full-range \
                 one; max |SR - full| = {moved:.3e} against scale {scale:.3e}"
            );

            // Same split as the scalar sweep: the doubled-root regime computes
            // SR as "full minus long range" over one shared `fac1`, so the
            // identity closes to round-off; above `rys_order = 3` it comes from
            // an INDEPENDENTLY constructed lower-bounded Gauss rule and closes
            // to that rule's accuracy instead.
            let rel_tol = if rys_order <= 3 { 1e-12 } else { 1e-10 };
            for idx in 0..n {
                let residual = (sr[idx] + lr[idx] - full[idx]).abs();
                assert!(
                    residual <= rel_tol * full[idx].abs().max(scale),
                    "{label} omega={omega} idx={idx}: SR({:.17e}) + LR({:.17e}) != \
                     full({:.17e}), residual={residual:.3e}",
                    sr[idx],
                    lr[idx],
                    full[idx]
                );
            }
            checked += 1;
        }

        // An explicit zero is the full-range operator, bit for bit.
        assert_eq!(
            eval_cintx(api, &shls, n, &atm, &bas, &env_with_omega(&env, 0.0))
                .unwrap_or_else(|e| panic!("{label} zero omega: {e:?}")),
            full,
            "{label}: env[8] = 0 must be byte-identical to an unset slot"
        );
    }

    assert!(
        checked >= 36,
        "every derivative row must be checked at both omegas, got {checked}"
    );
}

/// A row of the SAME families that `derivative_headroom` does not cover must
/// still refuse.
///
/// The GIAO/gauge rows read `env[8]` upstream just as `ip1` does, and their
/// launchers are host-routed too — so they would very likely just work. They are
/// refused anyway, because nothing gates them under a set ω, and a scope widened
/// past its gate is how a full-range substitute ships. The refusal is the
/// difference between "not yet verified" and "quietly wrong".
#[test]
fn a_row_outside_the_headroom_table_still_refuses_a_set_omega() {
    let (atm, bas, env) = build_spdf_fixture();
    let sr_env = env_with_omega(&env, -0.8);
    let shls = [0_i32, 4, 0, 4];
    let n = out_len(&shls, &bas) * 3;

    let err = eval_cintx(RawApiId::INT2E_G1_SPH, &shls, n, &atm, &bas, &sr_env)
        .expect_err("int2e_g1 under a set omega must refuse, not run full range");
    let text = format!("{err}");
    assert!(
        text.contains("range_omega"),
        "the refusal must name the parameter it is refusing, got: {text}"
    );
}

/// A family libcint itself ignores `env[8]` for must keep evaluating normally —
/// a caller inside a PySCF-style `range_coulomb(omega)` block leaves the slot
/// set for every integral it evaluates there. `int1e_ovlp` has no omega branch
/// anywhere in `g1e.c`.
#[test]
fn a_one_electron_operator_ignores_a_set_omega_exactly_as_libcint_does() {
    let (atm, bas, env) = build_spdf_fixture();
    let shls = [0_i32, 5];
    let n = out_len(&shls, &bas);

    let plain = eval_cintx(RawApiId::INT1E_OVLP_SPH, &shls, n, &atm, &bas, &env)
        .expect("int1e_ovlp with an unset omega");
    let with_omega = eval_cintx(
        RawApiId::INT1E_OVLP_SPH,
        &shls,
        n,
        &atm,
        &bas,
        &env_with_omega(&env, -0.8),
    )
    .expect("int1e_ovlp must ignore env[8], as libcint's 1e kernels do");
    assert_eq!(
        plain, with_omega,
        "int1e_ovlp must be unaffected by env[8]: libcint's g1e.c never reads it"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b. Spinor — `supports_range_omega` admits it, so it has to be covered
// ─────────────────────────────────────────────────────────────────────────────

/// Spinor length of a shell with `kappa = 0` and `nctr = 1`: `CINTlen_spinor`
/// returns `4l + 2` there (`cint_bas.c`), and the fixture leaves `KAPPA_OF`
/// unset. The `2 *` is the interleaved real/imaginary pair libcint writes.
fn spinor_out_len(shls: &[i32], bas: &[i32]) -> usize {
    2 * shls
        .iter()
        .map(|&s| (4 * bas[s as usize * BAS_SLOTS + ANG_OF] + 2) as usize)
        .product::<usize>()
}

/// `int2e_spinor` has `operator_name == "electron-repulsion"` and
/// `canonical_family == "2e"`, so `supports_range_omega` admits it: it falls
/// through `launch_two_electron_typed` to the same scalar section as
/// `int2e_sph`, routes host under a set ω, and applies the spinor transform
/// downstream of `cart_blocks`. That is right by construction — but "right by
/// construction" is what every silent substitution looks like from the inside,
/// and until D-PBC-24 P2-4 nothing exercised it.
///
/// The alternative was to narrow `supports_range_omega` to Cart/Spheric. This
/// test is why that was not needed.
#[test]
fn int2e_spinor_honours_a_set_omega_and_closes_sr_plus_lr() {
    let (atm, bas, env) = build_spdf_fixture();

    // (s,s|s,s) at rys_order 1 and (p,p|p,p) at rys_order 3 — both ends of the
    // doubled-root regime, on tuples that straddle both centres.
    for (label, shls) in [
        ("int2e_spinor(s,s|s,s)", vec![0_i32, 4, 0, 4]),
        ("int2e_spinor(p,p|p,p)", vec![1_i32, 5, 1, 5]),
    ] {
        let n = spinor_out_len(&shls, &bas);

        let full = eval_cintx(RawApiId::INT2E_SPINOR, &shls, n, &atm, &bas, &env)
            .unwrap_or_else(|e| panic!("{label} full range: {e:?}"));
        assert!(
            full.iter().any(|v| v.abs() > 1e-18),
            "{label}: the full-range spinor block is all zeros, so nothing below asserts anything"
        );

        for omega in [0.3_f64, 0.8] {
            let lr = eval_cintx(
                RawApiId::INT2E_SPINOR,
                &shls,
                n,
                &atm,
                &bas,
                &env_with_omega(&env, omega),
            )
            .unwrap_or_else(|e| panic!("{label} LR omega={omega}: {e:?}"));
            let sr = eval_cintx(
                RawApiId::INT2E_SPINOR,
                &shls,
                n,
                &atm,
                &bas,
                &env_with_omega(&env, -omega),
            )
            .unwrap_or_else(|e| panic!("{label} SR omega=-{omega}: {e:?}"));

            // ω must actually reach the kernel. A spinor path that dropped it
            // between `cart_blocks` and the transform would return the
            // full-range block here and pass every other assertion.
            let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let moved = (0..n).fold(0.0_f64, |m, i| m.max((sr[i] - full[i]).abs()));
            assert!(
                moved > 1e-3 * scale,
                "{label} omega=-{omega}: the spinor result is (almost) the full-range one; \
                 max |SR - full| = {moved:.3e} against scale {scale:.3e}"
            );

            // The transform is linear in the Cartesian block, so `SR + LR ==
            // full` survives it component by component, real and imaginary
            // alike.
            //
            // The floor is the BLOCK maximum, not the element's own magnitude:
            // the spinor transform is a dense mix of the Cartesian block, so it
            // produces elements that are ~1e-17 of the block through
            // cancellation. Those carry no information at f64, and holding them
            // to their own relative tolerance would gate on rounding noise.
            for idx in 0..n {
                let residual = (sr[idx] + lr[idx] - full[idx]).abs();
                assert!(
                    residual <= 1e-12 * full[idx].abs().max(scale),
                    "{label} omega={omega} idx={idx}: SR({:.17e}) + LR({:.17e}) != \
                     full({:.17e}), residual={residual:.3e}",
                    sr[idx],
                    lr[idx],
                    full[idx]
                );
            }
        }

        // And an explicit zero is still bit-for-bit the full-range operator.
        assert_eq!(
            eval_cintx(
                RawApiId::INT2E_SPINOR,
                &shls,
                n,
                &atm,
                &bas,
                &env_with_omega(&env, 0.0)
            )
            .unwrap_or_else(|e| panic!("{label} zero omega: {e:?}")),
            full,
            "{label}: env[8] = 0 must be byte-identical to an unset slot"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Vendor parity — the stage-2 acceptance gate
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
mod vendor {
    use super::*;
    use cintx_oracle::vendor_ffi;
    use cintx_oracle::vendor_ffi::{vendor_int2c2e_sph, vendor_int2e_sph, vendor_int3c2e_sph};

    fn eval_vendor(
        shls: &[i32],
        out_len: usize,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; out_len];
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        match shls.len() {
            2 => {
                let s: [i32; 2] = [shls[0], shls[1]];
                vendor_int2c2e_sph(&mut out, &s, atm, natm, bas, nbas, env);
            }
            3 => {
                let s: [i32; 3] = [shls[0], shls[1], shls[2]];
                vendor_int3c2e_sph(&mut out, &s, atm, natm, bas, nbas, env);
            }
            4 => {
                let s: [i32; 4] = [shls[0], shls[1], shls[2], shls[3]];
                vendor_int2e_sph(&mut out, &s, atm, natm, bas, nbas, env);
            }
            other => panic!("no scalar Coulomb symbol of arity {other}"),
        }
        out
    }

    /// cintx == vendored libcint 6.1.3 across the whole `env[8]` sweep.
    ///
    /// Tolerance, not byte identity: cintx's 2e/3c2e/2c2e kernels implement no
    /// `cceij` primitive screening — not for the full-range operator either —
    /// so they keep MORE primitives than upstream, which is the conservative
    /// direction (D-PBC-24 §3.5). The short-range `EXPCUTOFF_SR` guard IS
    /// ported, because it is a numerical-stability requirement rather than an
    /// optimisation.
    #[test]
    fn cintx_matches_vendored_libcint_across_the_omega_sweep() {
        let (atm, bas, env) = build_spdf_fixture();

        let mut compared = 0usize;
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();

        for (label, shls) in sweep_tuples() {
            let ls = tuple_ls(&shls);
            let order = rys_order(&ls);
            let api = api_for_arity(shls.len());
            let n = out_len(&shls, &bas);

            for omega in OMEGA_SWEEP {
                let env_omega = env_with_omega(&env, omega);
                let got = eval_cintx(api, &shls, n, &atm, &bas, &env_omega)
                    .unwrap_or_else(|e| panic!("{label} omega={omega}: {e:?}"));
                let want = eval_vendor(&shls, n, &atm, &bas, &env_omega);

                for idx in 0..n {
                    let diff = (got[idx] - want[idx]).abs();
                    let tol = 1e-12 + 1e-11 * want[idx].abs();
                    if diff > worst {
                        worst = diff;
                        worst_label = format!("{label} omega={omega} idx={idx}");
                    }
                    assert!(
                        diff <= tol,
                        "{label} rys_order={order} omega={omega} idx={idx}: \
                         cintx={:.17e} libcint={:.17e} diff={diff:.3e}",
                        got[idx],
                        want[idx]
                    );
                }
                compared += 1;
            }
        }

        assert!(
            compared >= 40,
            "the sweep must cover every supported (tuple, omega) pair, got {compared}"
        );
        eprintln!(
            "range_omega vendor parity: {compared} evaluations, worst |diff| = {worst:.3e} at {worst_label}"
        );
    }

    /// The WIDE sweep: exponents, separations and ω together, over the four
    /// families whose `rys_order` reaches the lower-bounded quadrature.
    ///
    /// `lower = sqrt(theta)` with `theta = ω²/(ω² + a0)`, so `lower` is set by
    /// the EXPONENTS and ω, and `x = a0 · rr` by the separation. Sweeping those
    /// three drives `lower` across essentially its whole `(0, 1)` range — the
    /// diffuse/large-ω corners reach `lower > 0.999` — and `x` from ~1e-3 to
    /// past the solver breakpoints. This is the test that says whether the
    /// stage-3 quadrature is fit for the integrals, as opposed to fit for a
    /// node-by-node comparison.
    ///
    /// Tolerance is relative to the block maximum: an individual integral that
    /// is 1e-9 of the largest in its block carries no information at f64.
    #[test]
    fn short_range_holds_across_a_wide_exponent_distance_and_omega_sweep() {
        // (label, l, shell tuple, output length, rys_order)
        let cases: [(&str, i32, &[i32], usize, usize); 4] = [
            ("int2c2e(f,f)", 3, &[0, 1], 49, 4),
            ("int3c2e(d,d|d)", 2, &[0, 1, 1], 125, 4),
            ("int2e(d,d|d,d)", 2, &[0, 1, 0, 1], 625, 5),
            ("int2e(f,f|f,f)", 3, &[0, 1, 0, 1], 2401, 7),
        ];

        let mut worst = 0.0_f64;
        let mut worst_at = String::new();
        let mut evaluated = 0usize;

        for (label, l, shls, n, order) in cases {
            let api = api_for_arity(shls.len());
            for &exp in &[0.02_f64, 0.15, 1.5, 5.0] {
                for &dist in &[0.5_f64, 4.0, 14.0] {
                    let (atm, bas, env) = build_two_centre_fixture(l, exp, dist);
                    for &omega in &[-0.05_f64, -0.5, -3.0, -8.0] {
                        let env_omega = env_with_omega(&env, omega);
                        let got =
                            eval_cintx(api, shls, n, &atm, &bas, &env_omega).unwrap_or_else(|e| {
                                panic!("{label} exp={exp} dist={dist} omega={omega}: {e:?}")
                            });
                        let want = eval_vendor(shls, n, &atm, &bas, &env_omega);
                        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                        if scale == 0.0 {
                            continue;
                        }
                        evaluated += 1;
                        for idx in 0..n {
                            let diff = (got[idx] - want[idx]).abs() / scale;
                            if diff > worst {
                                worst = diff;
                                worst_at = format!(
                                    "{label} rys_order={order} exp={exp} dist={dist} \
                                     omega={omega} idx={idx} block_max={scale:.2e}"
                                );
                            }
                            assert!(
                                diff <= 1e-8,
                                "{label} rys_order={order} exp={exp} dist={dist} omega={omega} \
                                 idx={idx}: cintx={:.17e} libcint={:.17e} \
                                 scaled diff={diff:.3e} (block max {scale:.3e})",
                                got[idx],
                                want[idx]
                            );
                        }
                    }
                }
            }
        }

        assert!(
            evaluated >= 150,
            "the wide sweep must actually evaluate, got {evaluated} blocks"
        );
        eprintln!(
            "range_omega wide SR sweep: {evaluated} blocks, worst scaled |diff| = {worst:.3e} \
             at {worst_at}"
        );
    }

    /// Derivative-row vendor parity across the ω sweep (D-PBC-24 P2-1).
    ///
    /// The `SR + LR == full` identity above is self-checking but blind to the
    /// CONVENTION: a row whose `ng[..INC]` raise landed on the wrong tuple
    /// position would satisfy it at every ω and disagree with libcint
    /// everywhere. This is the check that settles that, on the same tolerance
    /// as the scalar sweep.
    ///
    /// It is also the gate that licenses the widened
    /// `supports_range_omega`: every row admitted there appears here.
    #[test]
    fn cintx_matches_vendored_libcint_for_the_derivative_rows() {
        let (atm, bas, env) = build_spdf_fixture();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        // One vendor entry point per row, keyed by the same label.
        type V2 = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;
        type V3 = fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32;
        type V4 = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;

        fn vendor_for(
            label: &str,
            shls: &[i32],
            out: &mut [f64],
            atm: &[i32],
            natm: i32,
            bas: &[i32],
            nbas: i32,
            env: &[f64],
        ) {
            use cintx_oracle::vendor_ffi as v;
            match shls.len() {
                2 => {
                    let f: V2 = match label.split('(').next().unwrap() {
                        "int2c2e_ip1" => v::vendor_int2c2e_ip1_sph,
                        "int2c2e_ip2" => v::vendor_int2c2e_ip2_sph,
                        "int2c2e_ipip1" => v::vendor_int2c2e_ipip1_sph,
                        "int2c2e_ip1ip2" => v::vendor_int2c2e_ip1ip2_sph,
                        other => panic!("no vendor arity-2 entry point for {other}"),
                    };
                    f(out, &[shls[0], shls[1]], atm, natm, bas, nbas, env);
                }
                3 => {
                    let f: V3 = match label.split('(').next().unwrap() {
                        "int3c2e_ip1" => v::vendor_int3c2e_ip1_sph,
                        "int3c2e_ip2" => v::vendor_int3c2e_ip2_sph,
                        "int3c2e_ipip1" => v::vendor_int3c2e_ipip1_sph,
                        "int3c2e_ipip2" => v::vendor_int3c2e_ipip2_sph,
                        "int3c2e_ipvip1" => v::vendor_int3c2e_ipvip1_sph,
                        "int3c2e_ip1ip2" => v::vendor_int3c2e_ip1ip2_sph,
                        other => panic!("no vendor arity-3 entry point for {other}"),
                    };
                    f(out, &[shls[0], shls[1], shls[2]], atm, natm, bas, nbas, env);
                }
                4 => {
                    let f: V4 = match label.split('(').next().unwrap() {
                        "int2e_ip1" => v::vendor_int2e_ip1_sph,
                        "int2e_ip2" => v::vendor_int2e_ip2_sph,
                        "int2e_ipip1" => v::vendor_int2e_ipip1_sph,
                        "int2e_ipvip1" => v::vendor_int2e_ipvip1_sph,
                        "int2e_ip1ip2" => v::vendor_int2e_ip1ip2_sph,
                        other => panic!("no vendor arity-4 entry point for {other}"),
                    };
                    f(
                        out,
                        &[shls[0], shls[1], shls[2], shls[3]],
                        atm,
                        natm,
                        bas,
                        nbas,
                        env,
                    );
                }
                other => panic!("no scalar Coulomb derivative of arity {other}"),
            }
        }

        let mut compared = 0usize;
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();

        for r in derivative_rows() {
            let n = out_len(&r.shls, &bas) * r.ncomp;
            for omega in OMEGA_SWEEP {
                let env_omega = env_with_omega(&env, omega);
                let got = eval_cintx(r.api, &r.shls, n, &atm, &bas, &env_omega)
                    .unwrap_or_else(|e| panic!("{} omega={omega}: {e:?}", r.label));

                let mut want = vec![0.0_f64; n];
                vendor_for(
                    r.label, &r.shls, &mut want, &atm, natm, &bas, nbas, &env_omega,
                );

                for idx in 0..n {
                    let diff = (got[idx] - want[idx]).abs();
                    if diff > worst {
                        worst = diff;
                        worst_label = format!("{} omega={omega} idx={idx}", r.label);
                    }
                    assert!(
                        diff <= 1e-12 + 1e-11 * want[idx].abs(),
                        "{} rys_order={} omega={omega} idx={idx}: cintx={:.17e} \
                         libcint={:.17e} diff={diff:.3e}",
                        r.label,
                        r.rys_order,
                        got[idx],
                        want[idx]
                    );
                }
                compared += 1;
            }
        }

        assert_eq!(
            compared,
            derivative_rows().len() * OMEGA_SWEEP.len(),
            "every derivative row must be compared at every omega"
        );
        eprintln!(
            "range_omega derivative vendor parity: {compared} evaluations, \
             worst |diff| = {worst:.3e} at {worst_label}"
        );
    }

    /// Spinor vendor parity across the ω sweep (D-PBC-24 P2-4).
    ///
    /// `int2e_spinor` reaches the same `CINTg0_2e` omega branch as `int2e_sph`
    /// upstream, so the only thing that could differ on cintx's side is whether
    /// ω survives the interleaved-complex spinor transform. Comparing against
    /// the vendor is the only check that settles that, rather than settling
    /// that cintx agrees with itself.
    ///
    /// Same tolerance as the spherical sweep: cintx keeps more primitives than
    /// upstream (no `cceij` screening), so this is a tolerance rather than byte
    /// identity.
    #[test]
    fn cintx_matches_vendored_libcint_for_int2e_spinor_across_the_omega_sweep() {
        let (atm, bas, env) = build_spdf_fixture();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        let mut compared = 0usize;
        let mut worst = 0.0_f64;

        for (label, shls) in [
            ("int2e_spinor(s,s|s,s)", [0_i32, 4, 0, 4]),
            ("int2e_spinor(p,p|p,p)", [1_i32, 5, 1, 5]),
        ] {
            let n = spinor_out_len(&shls, &bas);
            for omega in OMEGA_SWEEP {
                let env_omega = env_with_omega(&env, omega);
                let got = eval_cintx(RawApiId::INT2E_SPINOR, &shls, n, &atm, &bas, &env_omega)
                    .unwrap_or_else(|e| panic!("{label} omega={omega}: {e:?}"));

                let mut want = vec![0.0_f64; n];
                vendor_ffi::vendor_int2e_spinor(
                    &mut want, &shls, &atm, natm, &bas, nbas, &env_omega,
                );

                for idx in 0..n {
                    let diff = (got[idx] - want[idx]).abs();
                    worst = worst.max(diff);
                    assert!(
                        diff <= 1e-12 + 1e-11 * want[idx].abs(),
                        "{label} omega={omega} idx={idx}: cintx={:.17e} libcint={:.17e} \
                         diff={diff:.3e}",
                        got[idx],
                        want[idx]
                    );
                }
                compared += 1;
            }
        }

        assert_eq!(compared, 10, "two tuples over the five-point omega sweep");
        eprintln!("range_omega spinor vendor parity: worst |diff| = {worst:.3e}");
    }

    /// The measurement D-PBC-24 stage 0 asks for: RAW per-tuple libcint 6.1.3
    /// values under the `env[8]` sweep, recorded BEFORE trusting any cintx
    /// number against them.
    ///
    /// Emits a stable, greppable table to stdout and — when
    /// `CINTX_RANGE_OMEGA_OUT` names a path — writes the same table there, so
    /// the committed `.out` is regenerated by running this test rather than by
    /// hand. Run with `--nocapture` to read it on the terminal.
    #[test]
    fn record_the_omega_sweep_reference_values() {
        let (atm, bas, env) = build_spdf_fixture();
        let mut lines = Vec::new();
        lines.push(
            "# D-PBC-24 stage 0 — vendored libcint 6.1.3 env[PTR_RANGE_OMEGA] sweep".to_owned(),
        );
        lines
            .push("# Fixture: two centres at (0,0,0) and (0.3,0,1.7); s/p/d/f on each,".to_owned());
        lines.push("#   one primitive and one contraction per shell, spherical form.".to_owned());
        lines.push(
            "# sr_regime: 'doubled_roots' = rys_order <= 3, where libcint computes SR as"
                .to_owned(),
        );
        lines.push(
            "#   full minus long range over 2*rys_order roots and NO new quadrature is".to_owned(),
        );
        lines.push(
            "#   needed; 'sr_rys_roots' = rys_order > 3, which needs the lower-bounded".to_owned(),
        );
        lines.push("#   CINTsr_rys_roots family (D-PBC-24 stage 3, not implemented).".to_owned());
        lines.push(String::new());
        lines.push(
            "tuple, rys_order, sr_regime, omega, n, max_abs, sum_abs, first, last".to_owned(),
        );

        for (label, shls) in sweep_tuples() {
            let order = rys_order(&tuple_ls(&shls));
            let regime = if order <= 3 {
                "doubled_roots"
            } else {
                "sr_rys_roots"
            };
            let n = out_len(&shls, &bas);
            for omega in OMEGA_SWEEP {
                let want = eval_vendor(&shls, n, &atm, &bas, &env_with_omega(&env, omega));
                let max_abs = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let sum_abs: f64 = want.iter().map(|v| v.abs()).sum();
                lines.push(format!(
                    "{label}, {order}, {regime}, {omega:+.1}, {n}, {max_abs:.17e}, \
                     {sum_abs:.17e}, {:.17e}, {:.17e}",
                    want[0],
                    want[n - 1]
                ));
            }
        }

        lines.push(String::new());
        lines.push("# SR(omega) + LR(omega) - full, worst absolute residual per tuple.".to_owned());
        lines.push("# This is the identity that catches an erf/erfc swap.".to_owned());
        lines.push("tuple, rys_order, omega, worst_residual".to_owned());
        for (label, shls) in sweep_tuples() {
            let order = rys_order(&tuple_ls(&shls));
            if order > 3 {
                continue;
            }
            let n = out_len(&shls, &bas);
            let full = eval_vendor(&shls, n, &atm, &bas, &env);
            for omega in [0.3_f64, 0.8] {
                let lr = eval_vendor(&shls, n, &atm, &bas, &env_with_omega(&env, omega));
                let sr = eval_vendor(&shls, n, &atm, &bas, &env_with_omega(&env, -omega));
                let worst = (0..n).fold(0.0_f64, |m, i| m.max((sr[i] + lr[i] - full[i]).abs()));
                lines.push(format!("{label}, {order}, {omega:+.1}, {worst:.3e}"));
            }
        }

        let table = lines.join("\n") + "\n";
        print!("{table}");
        if let Ok(path) = std::env::var("CINTX_RANGE_OMEGA_OUT") {
            std::fs::write(&path, &table).expect("write the stage-0 measurement record");
            eprintln!("wrote {path}");
        }
    }
}
