//! W4-06 — `intor2.c` gauge / cross-product 2e families, byte-identity vs vendored
//! libcint 6.1.3.
//!
//! `int2e_ip1v_r1`, `int2e_ip1v_rc1`, `int2e_ipvg1_xp1`, `int2e_ipvg2_xp1` — rank 9,
//! spin-free in both electrons (`ng[5] == ng[6] == 1`), so cart, sph AND spinor are
//! all real forms of the same family.
//!
//! The fixture is a `d`-shell `nctr=2` quartet with a NON-ZERO gauge origin. At
//! `ng = {1,2,0,0,...}` that is `nroots = (3+4+2+2)/2 + 1 = 6`, above
//! `MAX_DEVICE_NROOTS = 5`, so every case exercises the host Rys route.
//!
//! A non-zero `PTR_COMMON_ORIG` is essential: only `int2e_ip1v_rc1` reads it
//! (`G2E_RCJ`), while `ip1v_r1` uses a plain stride shift and both `xp1` families
//! raise about a BASIS CENTRE (`G2E_R0I` / `G2E_R0K`). With a zero origin the three
//! that must ignore it would pass even if they wrongly consumed it.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const ANG: i32 = 2;
const NCTR: usize = 2;
const RANK: usize = 9;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    // Non-zero gauge origin — see the module docs.
    env[PTR_COMMON_ORIG..PTR_COMMON_ORIG + 3].copy_from_slice(&[0.23, -0.41, 0.17]);
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

/// AO functions per shell in each representation (times `NCTR`).
fn shell_dim(rep: &str) -> usize {
    let l = ANG as usize;
    NCTR * match rep {
        "cart" => (l + 1) * (l + 2) / 2,
        "sph" => 2 * l + 1,
        "spinor" => 4 * l + 2,
        other => panic!("unknown representation {other}"),
    }
}

type VendorFn = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;

#[cfg(has_vendor_libcint)]
fn measure(symbol: &str, rep: &str, vendor_fn: VendorFn) -> f64 {
    let (atm, bas, env) = fixture();
    let shls = [0, 1, 2, 3];
    let d = shell_dim(rep);
    let complex = if rep == "spinor" { 2 } else { 1 };
    let len = RANK * d.pow(4) * complex;

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
        .unwrap();
    }
    assert_eq!(ours.len(), len, "{symbol}: component rank must be {RANK}");

    let mut reference = vec![0.0; len];
    assert_ne!(
        vendor_fn(&mut reference, &shls, &atm, 2, &bas, 4, &env),
        0,
        "{symbol}: vendored libcint returned an empty shell block"
    );
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
fn assert_all_green(label: &str, measured: &[(String, f64)]) {
    let mut report = String::new();
    let mut failed = 0usize;
    for (symbol, max_abs) in measured {
        let verdict = if *max_abs <= ATOL { "ok  " } else { "FAIL" };
        if *max_abs > ATOL {
            failed += 1;
        }
        report.push_str(&format!("\n  {verdict} {symbol:30} max_abs={max_abs:.3e}"));
    }
    assert!(
        failed == 0,
        "{label}: {failed}/{} cases exceed atol={ATOL:.0e}{report}",
        measured.len()
    );
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_gauge2e_cart_sph_spinor() {
    use cintx_oracle::vendor_ffi as v;
    let cases: [(&str, [VendorFn; 3]); 4] = [
        (
            "int2e_ip1v_r1",
            [
                v::vendor_int2e_ip1v_r1_cart,
                v::vendor_int2e_ip1v_r1_sph,
                v::vendor_int2e_ip1v_r1_spinor,
            ],
        ),
        (
            "int2e_ip1v_rc1",
            [
                v::vendor_int2e_ip1v_rc1_cart,
                v::vendor_int2e_ip1v_rc1_sph,
                v::vendor_int2e_ip1v_rc1_spinor,
            ],
        ),
        (
            "int2e_ipvg1_xp1",
            [
                v::vendor_int2e_ipvg1_xp1_cart,
                v::vendor_int2e_ipvg1_xp1_sph,
                v::vendor_int2e_ipvg1_xp1_spinor,
            ],
        ),
        (
            "int2e_ipvg2_xp1",
            [
                v::vendor_int2e_ipvg2_xp1_cart,
                v::vendor_int2e_ipvg2_xp1_sph,
                v::vendor_int2e_ipvg2_xp1_spinor,
            ],
        ),
    ];
    let mut measured = Vec::new();
    for (base, fns) in cases {
        for (rep, vendor_fn) in ["cart", "sph", "spinor"].into_iter().zip(fns) {
            let symbol = format!("{base}_{rep}");
            let max_abs = measure(&symbol, rep, vendor_fn);
            measured.push((symbol, max_abs));
        }
    }
    assert_all_green("gauge/cross-product 2e", &measured);
}

/// `int2e_ip1v_rc1` is the ONLY one of the four that reads `PTR_COMMON_ORIG`. Moving
/// the gauge origin must change its result and leave the other three untouched — the
/// direct guard against wiring the origin into the wrong families.
#[cfg(has_vendor_libcint)]
#[test]
fn only_rc1_depends_on_the_gauge_origin() {
    let (atm, bas, mut env) = fixture();
    let shls = [0, 1, 2, 3];
    let d = shell_dim("sph");
    let len = RANK * d.pow(4);

    let eval = |env: &[f64], symbol: &'static str| -> Vec<f64> {
        let mut out = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::Symbol(symbol),
                Some(&mut out),
                None,
                &shls,
                &atm,
                &bas,
                env,
                None,
                None,
            )
            .unwrap();
        }
        out
    };

    let symbols = [
        ("int2e_ip1v_r1_sph", false),
        ("int2e_ip1v_rc1_sph", true),
        ("int2e_ipvg1_xp1_sph", false),
        ("int2e_ipvg2_xp1_sph", false),
    ];
    let before: Vec<Vec<f64>> = symbols.iter().map(|(s, _)| eval(&env, s)).collect();

    env[PTR_COMMON_ORIG..PTR_COMMON_ORIG + 3].copy_from_slice(&[-0.6, 0.9, -0.15]);
    for ((symbol, sensitive), base) in symbols.iter().zip(&before) {
        let after = eval(&env, symbol);
        let max_delta = base
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if *sensitive {
            assert!(
                max_delta > 1e-8,
                "{symbol} must depend on PTR_COMMON_ORIG (G2E_RCJ) but is unchanged"
            );
        } else {
            assert!(
                max_delta == 0.0,
                "{symbol} must NOT read PTR_COMMON_ORIG, yet moving it changed the \
                 result by {max_delta:.3e}"
            );
        }
    }
}
