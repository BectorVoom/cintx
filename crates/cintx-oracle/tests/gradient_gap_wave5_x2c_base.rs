//! W5-06 — the X2C **base** families `int1e_pnucp` and `int1e_prinvp`,
//! byte-identity vs vendored libcint 6.1.3.
//!
//! Wave 3 shipped the derivatives of these two (`int1e_ippnucp`,
//! `int1e_ippnucpip`, `int1e_ipippnucp` and the rinv twins) and its gate claimed
//! the `pyscf/x2c` symbol set was satisfiable. It was not: `pyscf/x2c/x2c.py`
//! calls `int1e_pnucp` directly to build the X2C Hamiltonian itself, and only
//! `sfx2c1e_grad.py` / `sfx2c1e_hess.py` were actually covered.
//!
//! Both are `ng[] = {1, 1, 0, 0, 2, 1, 0, 1}` (intor1.c:990), rank 1, and share
//! one gout — the `∇i · ∇j` trace `s[0] + s[4] + s[8]` over the 2-leg table. They
//! differ only in the Coulomb-center list: `pnucp` sums over nuclei
//! (`CINT1e_drv(..., 2)`), `prinvp` uses the single `PTR_RINV_ORIG` center
//! (`CINT1e_drv(..., 1)`).
//!
//! `nroots = (li+1 + lj+1)/2 + 1 = 4` at `d`-shells — inside `MAX_DEVICE_NROOTS`,
//! but the family rides the host-routed deriv34 machine like its Wave-3
//! derivatives do, so the same path is exercised.
//!
//! The fixture carries a NON-ZERO rinv origin: with a zero origin `prinvp` would
//! agree with a wrongly-centred implementation.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const NCTR: usize = 2;
const RANK: usize = 1;

fn fixture(ang: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
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
    let mut bas = vec![0; 2 * BAS_SLOTS];
    for shell in 0..2 {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = shell as i32;
        bas[offset + ANG_OF] = ang;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = NCTR as i32;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn shell_dim(rep: &str, ang: i32) -> usize {
    let l = ang as usize;
    NCTR * match rep {
        "cart" => (l + 1) * (l + 2) / 2,
        "sph" => 2 * l + 1,
        "spinor" => 4 * l + 2,
        other => panic!("unknown representation {other}"),
    }
}

#[cfg(has_vendor_libcint)]
type VendorFn = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;

#[cfg(has_vendor_libcint)]
fn measure(symbol: &str, rep: &str, ang: i32, vendor_fn: VendorFn) -> f64 {
    let (atm, bas, env) = fixture(ang);
    let shls = [0_i32, 1];
    let d = shell_dim(rep, ang);
    let complex = if rep == "spinor" { 2 } else { 1 };
    let len = RANK * d * d * complex;

    let symbol_static: &'static str = Box::leak(symbol.to_owned().into_boxed_str());
    let mut ours = vec![0.0; len];
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol_static),
            Some(&mut ours),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .unwrap_or_else(|e| panic!("{symbol}: cintx evaluation failed: {e}"));
    assert_eq!(ours.len(), len, "{symbol}: component rank must be {RANK}");

    let mut reference = vec![0.0; len];
    vendor_fn(&mut reference, &shls, &atm, 2, &bas, 2, &env);
    assert!(
        reference.iter().any(|v| v.abs() > 1e-14),
        "{symbol}: vendor reference is all-zero (driver not linked)"
    );

    ours.iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_x2c_base_families_all_reps() {
    use cintx_oracle::vendor_ffi as v;

    // The spinor forms are included: `CINT1e_spinor_drv` is a real driver, so
    // unlike the 2c2e/3c1e spinor rows this wave met, these ARE provable.
    let cases: [(&str, [VendorFn; 3]); 2] = [
        (
            "int1e_pnucp",
            [
                v::vendor_int1e_pnucp_cart,
                v::vendor_int1e_pnucp_sph,
                v::vendor_int1e_pnucp_spinor,
            ],
        ),
        (
            "int1e_prinvp",
            [
                v::vendor_int1e_prinvp_cart,
                v::vendor_int1e_prinvp_sph,
                v::vendor_int1e_prinvp_spinor,
            ],
        ),
    ];

    let mut measured: Vec<(String, f64)> = Vec::new();
    // s, p and d shells: `nroots` moves 2 → 3 → 4 across them.
    for ang in [0_i32, 1, 2] {
        for (symbol, fns) in cases {
            for (rep, vendor_fn) in ["cart", "sph", "spinor"].iter().zip(fns) {
                let residual = measure(&format!("{symbol}_{rep}"), rep, ang, vendor_fn);
                measured.push((format!("{symbol}_{rep} l={ang}"), residual));
            }
        }
    }

    let mut report = String::new();
    let mut failed = 0usize;
    for (label, residual) in &measured {
        let verdict = if *residual <= ATOL { "ok  " } else { "FAIL" };
        if *residual > ATOL {
            failed += 1;
        }
        report.push_str(&format!("\n  {verdict} {label:28} max_abs={residual:.3e}"));
    }
    assert!(
        failed == 0,
        "W5-06 X2C base families: {failed}/{} cases exceed atol={ATOL:.0e}{report}",
        measured.len()
    );
}

/// The rinv family must actually READ the origin: moving it has to change
/// `prinvp` and leave `pnucp` bit-identical. Catches a wrongly-centred or
/// silently-ignored `PTR_RINV_ORIG`, and needs no vendor build.
#[test]
fn only_prinvp_depends_on_the_rinv_origin() {
    let eval = |symbol: &'static str, origin: [f64; 3]| -> Vec<f64> {
        let (atm, bas, mut env) = fixture(2);
        env[PTR_RINV_ORIG..PTR_RINV_ORIG + 3].copy_from_slice(&origin);
        let d = shell_dim("sph", 2);
        let mut out = vec![0.0; RANK * d * d];
        let shls = [0_i32, 1];
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
        .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        out
    };

    let moved = [0.37_f64, -0.15, 0.62];
    let base = [-0.11_f64, 0.29, 0.05];

    let prinvp_a = eval("int1e_prinvp_sph", base);
    let prinvp_b = eval("int1e_prinvp_sph", moved);
    assert_ne!(
        prinvp_a, prinvp_b,
        "int1e_prinvp must depend on PTR_RINV_ORIG — it is the per-nucleus form"
    );

    let pnucp_a = eval("int1e_pnucp_sph", base);
    let pnucp_b = eval("int1e_pnucp_sph", moved);
    assert_eq!(
        pnucp_a, pnucp_b,
        "int1e_pnucp is atom-summed and must ignore PTR_RINV_ORIG entirely"
    );
}
