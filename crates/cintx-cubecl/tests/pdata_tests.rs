//! PairData validation tests against hand-computed references.
//!
//! All golden values cite the libcint source formula they validate:
//!   - zeta_ab = ai + aj           — g1e.c line 130
//!   - center_p = (ai*Ri + aj*Rj) / zeta_ab — g1e.c lines 131-133
//!   - rirj = Ri - Rj              — g1e.c line 135
//!   - fac = exp(-ai*aj/zeta_ab * |rirj|^2) * norm_i * norm_j — g1e.c line 134
//!   - aij2 = 0.5 / zeta_ab       — g1e.c line 168
//!
//! All assertions use absolute tolerance 1e-12 per design decision D-19.

use cintx_cubecl::math::pdata::compute_pdata_host;

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: Equal exponents — symmetric H2 geometry
// ai=1.0, aj=1.0, Ri=(0,0,0), Rj=(1.4,0,0), norm_i=norm_j=1.0
// Golden values:
//   zeta_ab = 1.0 + 1.0 = 2.0                   — g1e.c line 130
//   center_p = (1.0*0 + 1.0*1.4) / 2.0 = (0.7, 0, 0)  — g1e.c lines 131-133
//   rirj = (0-1.4, 0, 0) = (-1.4, 0, 0)          — g1e.c line 135
//   rr = 1.4^2 = 1.96
//   fac = exp(-1.0*1.0/2.0 * 1.96) * 1.0 = exp(-0.98) — g1e.c line 134
//   aij2 = 0.5 / 2.0 = 0.25                      — g1e.c line 168
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn pdata_equal_exponents() {
    let p = compute_pdata_host(
        1.0, 1.0,
        0.0, 0.0, 0.0,
        1.4, 0.0, 0.0,
        1.0, 1.0,
    );

    // zeta_ab = ai + aj, g1e.c line 130
    let expected_zeta = 2.0f64;
    assert!(
        (p.zeta_ab - expected_zeta).abs() < 1e-12,
        "zeta_ab: got {}, expected {expected_zeta}",
        p.zeta_ab
    );

    // center_p_x = (ai*Ri_x + aj*Rj_x) / zeta_ab = (0+1.4)/2.0 = 0.7, g1e.c line 131
    let expected_cpx = 0.7f64;
    assert!(
        (p.center_p_x - expected_cpx).abs() < 1e-12,
        "center_p_x: got {}, expected {expected_cpx}",
        p.center_p_x
    );
    assert!(
        p.center_p_y.abs() < 1e-12,
        "center_p_y: got {}, expected 0.0",
        p.center_p_y
    );
    assert!(
        p.center_p_z.abs() < 1e-12,
        "center_p_z: got {}, expected 0.0",
        p.center_p_z
    );

    // rirj = Ri - Rj, g1e.c line 135
    let expected_rirj_x = -1.4f64;
    assert!(
        (p.rirj_x - expected_rirj_x).abs() < 1e-12,
        "rirj_x: got {}, expected {expected_rirj_x}",
        p.rirj_x
    );

    // fac = exp(-ai*aj/zeta_ab * rr) * norm_i * norm_j = exp(-0.98), g1e.c line 134
    let expected_fac = (-0.98f64).exp();
    assert!(
        (p.fac - expected_fac).abs() < 1e-12,
        "fac: got {}, expected {expected_fac}",
        p.fac
    );

    // aij2 = 0.5 / zeta_ab = 0.25, g1e.c line 168
    let expected_aij2 = 0.25f64;
    assert!(
        (p.aij2 - expected_aij2).abs() < 1e-12,
        "aij2: got {}, expected {expected_aij2}",
        p.aij2
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: Asymmetric exponents — center shifts toward heavier exponent
// ai=5.0, aj=2.0, Ri=(0,0,0), Rj=(1,0,0), norm_i=norm_j=1.0
// Golden values:
//   zeta_ab = 5.0 + 2.0 = 7.0                    — g1e.c line 130
//   center_p_x = (5.0*0 + 2.0*1) / 7.0 = 2/7     — g1e.c line 131
//   rirj = (0-1, 0, 0) = (-1, 0, 0)               — g1e.c line 135
//   rr = 1.0
//   fac = exp(-5.0*2.0/7.0 * 1.0) = exp(-10/7)    — g1e.c line 134
//   aij2 = 0.5 / 7.0                              — g1e.c line 168
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn pdata_asymmetric() {
    let p = compute_pdata_host(
        5.0, 2.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0,
    );

    // zeta_ab = 7.0, g1e.c line 130
    assert!((p.zeta_ab - 7.0).abs() < 1e-12, "zeta_ab: got {}", p.zeta_ab);

    // center_p_x = 2/7 ≈ 0.285714... — center shifts toward Ri (ai=5 dominates)
    // g1e.c line 131: center_p = (ai*Ri + aj*Rj) / zeta_ab
    let expected_cpx = 2.0f64 / 7.0;
    assert!(
        (p.center_p_x - expected_cpx).abs() < 1e-12,
        "center_p_x: got {}, expected {expected_cpx}",
        p.center_p_x
    );

    // fac = exp(-10/7) * 1.0, g1e.c line 134
    let expected_fac = (-10.0f64 / 7.0).exp();
    assert!(
        (p.fac - expected_fac).abs() < 1e-12,
        "fac: got {}, expected {expected_fac}",
        p.fac
    );

    // aij2 = 0.5/7.0, g1e.c line 168
    let expected_aij2 = 0.5f64 / 7.0;
    assert!(
        (p.aij2 - expected_aij2).abs() < 1e-12,
        "aij2: got {}, expected {expected_aij2}",
        p.aij2
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: Coincident centers — no exponential decay
// ai=3.0, aj=4.0, Ri=Rj=(1,2,3), norm_i=norm_j=1.0
// Golden values:
//   zeta_ab = 7.0                                 — g1e.c line 130
//   center_p = Ri = (1, 2, 3)                     — g1e.c lines 131-133
//   rirj = (0, 0, 0)                              — g1e.c line 135
//   rr = 0.0
//   fac = exp(0) * 1.0 = 1.0                     — g1e.c line 134
//   aij2 = 0.5/7.0                               — g1e.c line 168
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn pdata_coincident() {
    let p = compute_pdata_host(
        3.0, 4.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 1.0,
    );

    // zeta_ab = 7.0, g1e.c line 130
    assert!((p.zeta_ab - 7.0).abs() < 1e-12, "zeta_ab: got {}", p.zeta_ab);

    // center_p = Ri = (1, 2, 3) when Ri == Rj, g1e.c lines 131-133
    assert!((p.center_p_x - 1.0).abs() < 1e-12, "center_p_x: got {}", p.center_p_x);
    assert!((p.center_p_y - 2.0).abs() < 1e-12, "center_p_y: got {}", p.center_p_y);
    assert!((p.center_p_z - 3.0).abs() < 1e-12, "center_p_z: got {}", p.center_p_z);

    // rirj = (0, 0, 0), g1e.c line 135
    assert!(p.rirj_x.abs() < 1e-12, "rirj_x: got {}", p.rirj_x);
    assert!(p.rirj_y.abs() < 1e-12, "rirj_y: got {}", p.rirj_y);
    assert!(p.rirj_z.abs() < 1e-12, "rirj_z: got {}", p.rirj_z);

    // fac = exp(0) * norm_i * norm_j = 1.0, g1e.c line 134
    assert!(
        (p.fac - 1.0).abs() < 1e-12,
        "fac: got {}, expected 1.0",
        p.fac
    );

    // aij2 = 0.5/7.0, g1e.c line 168
    let expected_aij2 = 0.5f64 / 7.0;
    assert!(
        (p.aij2 - expected_aij2).abs() < 1e-12,
        "aij2: got {}, expected {expected_aij2}",
        p.aij2
    );
}
