//! Integration tests for the cart-to-sph (c2s) transform module.
//!
//! Validates Condon-Shortley coefficient matrices against libcint cart2sph.c
//! and verifies transform correctness for key shell pairs.

use cintx_cubecl::transform::c2s::{
    cart_to_sph_1e, cart_to_spheric_staging, ncart, nsph, C2S_L0, C2S_L1, C2S_L2, C2S_L3,
    C2S_L4,
};

// ──────────────────────────────────────────────────────────────────────────
//  Helper
// ──────────────────────────────────────────────────────────────────────────

fn assert_approx_eq(a: f64, b: f64, atol: f64, label: &str) {
    let diff = (a - b).abs();
    assert!(diff <= atol, "{label}: |{a} - {b}| = {diff} > {atol}");
}

// ──────────────────────────────────────────────────────────────────────────
//  Dimension helpers
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn test_ncart_values() {
    assert_eq!(ncart(0), 1);
    assert_eq!(ncart(1), 3);
    assert_eq!(ncart(2), 6);
    assert_eq!(ncart(3), 10);
    assert_eq!(ncart(4), 15);
}

#[test]
fn test_nsph_values() {
    assert_eq!(nsph(0), 1);
    assert_eq!(nsph(1), 3);
    assert_eq!(nsph(2), 5);
    assert_eq!(nsph(3), 7);
    assert_eq!(nsph(4), 9);
}

// ──────────────────────────────────────────────────────────────────────────
//  Coefficient matrix validation
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn test_c2s_l0_identity() {
    assert_eq!(C2S_L0, [[1.0]]);
    assert_eq!(C2S_L0.len(), 1);
    assert_eq!(C2S_L0[0].len(), 1);
}

#[test]
fn test_c2s_l1_coefficients() {
    // p-shell: 3x3 identity matrix.
    //
    // Libcint uses (px, py, pz) ordering for the p-shell spherical components —
    // NOT the standard spherical harmonic (py, pz, px) = (m=-1, m=0, m=+1) ordering.
    // The CINTcommon_fac_sp(1) normalization (0.4886) is applied in the primitive loop
    // rather than embedded in these coefficients, so C2S_L1 is the identity transform.
    // Reference: libcint cart2sph.c `g_trans_cart2sph[]` p-shell section.
    assert_eq!(C2S_L1.len(), 3, "L1 must have 3 rows (2*1+1)");
    assert_eq!(C2S_L1[0].len(), 3, "L1 must have 3 cols (ncart(1))");

    // sph[0] = px: only x component active
    assert_approx_eq(C2S_L1[0][0], 1.0, 1e-15, "L1[sph0][x] = 1 (px)");
    assert_approx_eq(C2S_L1[0][1], 0.0, 1e-15, "L1[sph0][y] = 0");
    assert_approx_eq(C2S_L1[0][2], 0.0, 1e-15, "L1[sph0][z] = 0");

    // sph[1] = py: only y component active
    assert_approx_eq(C2S_L1[1][0], 0.0, 1e-15, "L1[sph1][x] = 0");
    assert_approx_eq(C2S_L1[1][1], 1.0, 1e-15, "L1[sph1][y] = 1 (py)");
    assert_approx_eq(C2S_L1[1][2], 0.0, 1e-15, "L1[sph1][z] = 0");

    // sph[2] = pz: only z component active
    assert_approx_eq(C2S_L1[2][0], 0.0, 1e-15, "L1[sph2][x] = 0");
    assert_approx_eq(C2S_L1[2][1], 0.0, 1e-15, "L1[sph2][y] = 0");
    assert_approx_eq(C2S_L1[2][2], 1.0, 1e-15, "L1[sph2][z] = 1 (pz)");
}

#[test]
fn test_c2s_l2_coefficients() {
    assert_eq!(C2S_L2.len(), 5, "L2 must have 5 rows (2*2+1)");
    assert_eq!(C2S_L2[0].len(), 6, "L2 must have 6 cols (ncart(2))");

    // m=-2 (dxy): C2S_L2[0][1] = 1.092548430592079070
    assert_approx_eq(C2S_L2[0][1], 1.092548430592079070, 1e-15, "L2[m=-2][xy]");

    // m=0 (dz2): C2S_L2[2][0] = -0.315391565252520002 (xx coeff)
    assert_approx_eq(C2S_L2[2][0], -0.315391565252520002, 1e-15, "L2[m=0][xx]");

    // m=0 (dz2): C2S_L2[2][5] = 0.630783130505040012 (zz coeff)
    assert_approx_eq(C2S_L2[2][5], 0.630783130505040012, 1e-15, "L2[m=0][zz]");

    // m=+2 (dx2-y2): C2S_L2[4][0] = 0.546274215296039535 (xx coeff)
    assert_approx_eq(C2S_L2[4][0], 0.546274215296039535, 1e-15, "L2[m=+2][xx]");

    // m=+2 (dx2-y2): C2S_L2[4][3] = -0.546274215296039535 (yy coeff)
    assert_approx_eq(C2S_L2[4][3], -0.546274215296039535, 1e-15, "L2[m=+2][yy]");

    // All-zero positions in m=-2 row
    assert_approx_eq(C2S_L2[0][0], 0.0, 1e-15, "L2[m=-2][xx] should be 0");
    assert_approx_eq(C2S_L2[0][2], 0.0, 1e-15, "L2[m=-2][xz] should be 0");
    assert_approx_eq(C2S_L2[0][3], 0.0, 1e-15, "L2[m=-2][yy] should be 0");
    assert_approx_eq(C2S_L2[0][4], 0.0, 1e-15, "L2[m=-2][yz] should be 0");
    assert_approx_eq(C2S_L2[0][5], 0.0, 1e-15, "L2[m=-2][zz] should be 0");
}

#[test]
fn test_c2s_l3_dimensions() {
    assert_eq!(C2S_L3.len(), 7, "L3 must have 7 rows (2*3+1)");
    assert_eq!(C2S_L3[0].len(), 10, "L3 must have 10 cols (ncart(3))");

    // Spot-check m=-3 (fyx2): offset 40 in g_trans_cart2sph
    // C2S_L3[0][1] = 1.770130769779930531
    assert_approx_eq(C2S_L3[0][1], 1.770130769779930531, 1e-15, "L3[m=-3][1]");
    // C2S_L3[0][6] = -0.590043589926643510
    assert_approx_eq(C2S_L3[0][6], -0.590043589926643510, 1e-15, "L3[m=-3][6]");

    // m=+3 (fx3): C2S_L3[6][0] = 0.590043589926643510
    assert_approx_eq(C2S_L3[6][0], 0.590043589926643510, 1e-15, "L3[m=+3][0]");
    // C2S_L3[6][3] = -1.770130769779930530
    assert_approx_eq(C2S_L3[6][3], -1.770130769779930530, 1e-15, "L3[m=+3][3]");
}

#[test]
fn test_c2s_l4_dimensions() {
    assert_eq!(C2S_L4.len(), 9, "L4 must have 9 rows (2*4+1)");
    assert_eq!(C2S_L4[0].len(), 15, "L4 must have 15 cols (ncart(4))");

    // Spot-check m=-4 (gyx3): C2S_L4[0][1] = 2.503342941796704538
    assert_approx_eq(C2S_L4[0][1], 2.503342941796704538, 1e-15, "L4[m=-4][1]");

    // m=0 (gz4): C2S_L4[4][14] = 0.846284375321634430 (zzzz)
    assert_approx_eq(C2S_L4[4][14], 0.846284375321634430, 1e-15, "L4[m=0][14]");

    // m=+4 (gy4): C2S_L4[8][0] = 0.625835735449176134 (xxxx)
    assert_approx_eq(C2S_L4[8][0], 0.625835735449176134, 1e-15, "L4[m=+4][0]");
}

// ──────────────────────────────────────────────────────────────────────────
//  Transform function correctness
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn test_c2s_ss_identity() {
    // s-s shell pair: 1x1 -> 1x1, identity
    let cart_buf = [1.0_f64];
    let mut sph_buf = [0.0_f64];
    cart_to_sph_1e(&cart_buf, &mut sph_buf, 0, 0);
    assert_approx_eq(sph_buf[0], 1.0, 1e-15, "ss identity");
}

#[test]
fn test_c2s_pp_transform() {
    // p-p shell pair: 3 cart x 3 cart -> 3 sph x 3 sph
    // Input: 9-element cart_buf (j=outer, i=inner)
    // Layout: cart_buf[j * nci + ci] with nci=3, ncj=3
    // Use identity matrix as input: cart_buf[j][i] = 1 if i==j else 0
    let cart_buf: [f64; 9] = [
        1.0, 0.0, 0.0, // j=0 (xx)
        0.0, 1.0, 0.0, // j=1 (xy)
        0.0, 0.0, 1.0, // j=2 (xz)
    ];
    let mut sph_buf = [0.0_f64; 9];
    cart_to_sph_1e(&cart_buf, &mut sph_buf, 1, 1);

    // Output shape is 3x3 (nsph(1)=3 for both i and j)
    // Verify it's not all zeros
    let nonzero_count = sph_buf.iter().filter(|&&x| x.abs() > 1e-15).count();
    assert!(nonzero_count > 0, "pp transform should produce non-zero output");

    // For identity cart input, C2S_L1 is itself a permutation matrix,
    // so the output should also be a permutation matrix (each row has exactly one 1.0)
    // Check each row of sph output has exactly one non-zero entry equal to 1.0
    for mi in 0..3 {
        let row_start = mi * 3;
        let row = &sph_buf[row_start..row_start + 3];
        let ones: Vec<_> = row.iter().filter(|&&x| (x - 1.0).abs() < 1e-15).collect();
        let zeros: Vec<_> = row.iter().filter(|&&x| x.abs() < 1e-15).collect();
        assert_eq!(ones.len(), 1, "pp output row {mi} should have exactly one 1.0");
        assert_eq!(zeros.len(), 2, "pp output row {mi} should have exactly two 0.0s");
    }
}

#[test]
fn test_c2s_ds_transform() {
    // d-s shell pair: 6 cart -> 5 sph (bra=d, ket=s)
    // Input: 6 elements (ncart(2)=6, ncart(0)=1)
    // Layout: cart_buf[j * nci + ci] with nci=6, ncj=1
    let cart_buf: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // pure xx component
    let mut sph_buf = [0.0_f64; 5]; // nsph(2)*nsph(0) = 5*1
    cart_to_sph_1e(&cart_buf, &mut sph_buf, 2, 0);

    // xx input projects onto:
    //   m=-2 (dxy):  C2S_L2[0][0] = 0.0
    //   m=-1 (dyz):  C2S_L2[1][0] = 0.0
    //   m= 0 (dz2):  C2S_L2[2][0] = -0.315391565252520002
    //   m=+1 (dxz):  C2S_L2[3][0] = 0.0
    //   m=+2 (dx2y2): C2S_L2[4][0] = 0.546274215296039535
    assert_approx_eq(sph_buf[0], 0.0, 1e-15, "ds[m=-2]");
    assert_approx_eq(sph_buf[1], 0.0, 1e-15, "ds[m=-1]");
    assert_approx_eq(sph_buf[2], -0.315391565252520002, 1e-15, "ds[m=0]");
    assert_approx_eq(sph_buf[3], 0.0, 1e-15, "ds[m=+1]");
    assert_approx_eq(sph_buf[4], 0.546274215296039535, 1e-15, "ds[m=+2]");
}

#[test]
fn test_c2s_sd_transform() {
    // s-d shell pair: 6 cart -> 5 sph (bra=s, ket=d)
    // Input: 6 elements (ncart(0)=1, ncart(2)=6)
    // Layout: cart_buf[j * nci + ci] with nci=1, ncj=6
    // Only the zz component active (j=5)
    let mut cart_buf = [0.0_f64; 6];
    cart_buf[5] = 1.0; // zz component
    let mut sph_buf = [0.0_f64; 5]; // nsph(0)*nsph(2) = 1*5
    cart_to_sph_1e(&cart_buf, &mut sph_buf, 0, 2);

    // zz input (j=5) projects onto ket sph components:
    //   m=-2: C2S_L2[0][5] = 0.0
    //   m=-1: C2S_L2[1][5] = 0.0
    //   m= 0: C2S_L2[2][5] = 0.630783130505040012
    //   m=+1: C2S_L2[3][5] = 0.0
    //   m=+2: C2S_L2[4][5] = 0.0
    assert_approx_eq(sph_buf[0], 0.0, 1e-15, "sd[mj=-2]");
    assert_approx_eq(sph_buf[1], 0.0, 1e-15, "sd[mj=-1]");
    assert_approx_eq(sph_buf[2], 0.630783130505040012, 1e-15, "sd[mj=0]");
    assert_approx_eq(sph_buf[3], 0.0, 1e-15, "sd[mj=+1]");
    assert_approx_eq(sph_buf[4], 0.0, 1e-15, "sd[mj=+2]");
}

#[test]
fn test_cart_to_spheric_staging_noop() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let original = data.clone();
    cart_to_spheric_staging(&mut data).expect("staging no-op should not fail");
    assert_eq!(data, original, "cart_to_spheric_staging must not modify data");
}

#[test]
fn test_cart_to_spheric_staging_noop_empty() {
    let mut data: Vec<f64> = vec![];
    cart_to_spheric_staging(&mut data).expect("empty staging should succeed");
}
