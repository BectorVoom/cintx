//! Obara-Saika recurrence unit tests.
//!
//! Tests `vrr_step`, `hrr_step`, and `vrr_2e_step` from `cintx_cubecl::math::obara_saika`.
//!
//! All assertions use 1e-12 atol per D-19. Tests use host-side wrappers
//! (`*_host` functions) to avoid CubeCL kernel launch machinery in unit tests.
//!
//! Reference: libcint-master/src/g1e.c lines 164-182 (VRR), 175-182 (HRR).

use approx::assert_abs_diff_eq;
use cintx_cubecl::math::obara_saika::{hrr_step_host, vrr_2e_step_host, vrr_step_host};

// ─────────────────────────────────────────────────────────────────────────────
//  vrr_step tests
// ─────────────────────────────────────────────────────────────────────────────

/// VRR with nmax=0 (s-shell): no recurrence needed; g-array unchanged after call.
///
/// vrr_step requires nmax >= 1 to build g[1]; for nmax=0 the function is a no-op.
#[test]
fn os_vrr_s_shell() {
    let mut g = vec![1.0f64; 4];
    // nmax=0: VRR does nothing — no iterations
    vrr_step_host(&mut g, 0.5, 0.25, 0, 1);
    // g[0] must be unchanged
    assert_abs_diff_eq!(g[0], 1.0, epsilon = 1e-12);
}

/// VRR p-shell (nmax=1): g[1] = rijrx * g[0].
///
/// g[0]=1.0, rijrx=0.5 -> g[1] = 0.5 * 1.0 = 0.5
#[test]
fn os_vrr_p_shell() {
    let mut g = vec![0.0f64; 4];
    g[0] = 1.0;
    vrr_step_host(&mut g, 0.5, 0.25, 1, 1);
    assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
}

/// VRR d-shell (nmax=2): g[1] and g[2].
///
/// g[0]=1.0, rijrx=0.5, aij2=0.25:
///   g[1] = 0.5 * 1.0 = 0.5
///   g[2] = 1 * 0.25 * g[0] + 0.5 * g[1] = 0.25 + 0.25 = 0.5
#[test]
fn os_vrr_d_shell() {
    let mut g = vec![0.0f64; 4];
    g[0] = 1.0;
    vrr_step_host(&mut g, 0.5, 0.25, 2, 1);
    assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(g[2], 0.5, epsilon = 1e-12);
}

/// VRR f-shell (nmax=3): g[3] = 2*aij2*g[1] + rijrx*g[2].
///
/// g[0]=1.0, rijrx=0.5, aij2=0.25:
///   g[1] = 0.5
///   g[2] = 0.5
///   g[3] = 2 * 0.25 * 0.5 + 0.5 * 0.5 = 0.25 + 0.25 = 0.5
#[test]
fn os_vrr_f_shell() {
    let mut g = vec![0.0f64; 8];
    g[0] = 1.0;
    vrr_step_host(&mut g, 0.5, 0.25, 3, 1);
    assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(g[2], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(g[3], 0.5, epsilon = 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
//  hrr_step tests
// ─────────────────────────────────────────────────────────────────────────────

/// HRR basic (lj=1): g[j=1,i=0] = g[j=0,i=1] + rirj*g[j=0,i=0].
///
/// Layout: stride di=1, dj=4 (simulating 4 i-slots per j-row)
/// g[0*4 + 0*1] = g[i=0,j=0] (set to 2.0)
/// g[0*4 + 1*1] = g[i=1,j=0] (set to 3.0)
/// After HRR: g[1*4 + 0*1] = g[0*4 + 1*1] + rirj * g[0*4 + 0*1]
///          = 3.0 + 1.5 * 2.0 = 6.0
#[test]
fn os_hrr_basic() {
    let mut g = vec![0.0f64; 16];
    // di=1 (i-stride), dj=4 (j-stride)
    // j=0 row: g[0]=2.0 (i=0,j=0), g[1]=3.0 (i=1,j=0)
    g[0] = 2.0; // g[j=0, i=0]
    g[1] = 3.0; // g[j=0, i=1]
    let rirj = 1.5_f64;
    let di = 1_u32;
    let dj = 4_u32;
    // lj=1, li_max=1 (so i can be 0 after j transfer)
    hrr_step_host(&mut g, rirj, di, dj, 1, 1);
    // g[j=1, i=0] = g[j=0, i=1] + rirj * g[j=0, i=0]
    //             = 3.0 + 1.5 * 2.0 = 6.0
    assert_abs_diff_eq!(g[4], 6.0, epsilon = 1e-12);
}

/// HRR d-transfer (lj=2): second HRR step builds from first.
///
/// Layout: di=1, dj=4, li_max=2
/// Initial j=0 row: g[i=0]=1.0, g[i=1]=2.0, g[i=2]=3.0
/// rirj=1.0
///
/// After j=1 pass:
///   g[4+0] = g[0+1] + 1.0*g[0+0] = 2.0 + 1.0 = 3.0
///   g[4+1] = g[0+2] + 1.0*g[0+1] = 3.0 + 2.0 = 5.0
///
/// After j=2 pass:
///   g[8+0] = g[4+1] + 1.0*g[4+0] = 5.0 + 3.0 = 8.0
#[test]
fn os_hrr_d_transfer() {
    let mut g = vec![0.0f64; 16];
    g[0] = 1.0; // g[j=0, i=0]
    g[1] = 2.0; // g[j=0, i=1]
    g[2] = 3.0; // g[j=0, i=2]
    hrr_step_host(&mut g, 1.0, 1, 4, 2, 2);
    assert_abs_diff_eq!(g[4], 3.0, epsilon = 1e-12); // g[j=1, i=0]
    assert_abs_diff_eq!(g[5], 5.0, epsilon = 1e-12); // g[j=1, i=1]
    assert_abs_diff_eq!(g[8], 8.0, epsilon = 1e-12); // g[j=2, i=0]
}

// ─────────────────────────────────────────────────────────────────────────────
//  vrr_2e_step tests
// ─────────────────────────────────────────────────────────────────────────────

/// vrr_2e_step basic (nmax=2, c00=0.3, b10=0.2):
///
/// g[0]=1.0 (base case)
/// g[1] = c00 * g[0] = 0.3
/// g[2] = 1 * b10 * g[0] + c00 * g[1] = 0.2*1.0 + 0.3*0.3 = 0.2 + 0.09 = 0.29
#[test]
fn os_vrr_2e_basic() {
    let mut g = vec![0.0f64; 8];
    g[0] = 1.0;
    vrr_2e_step_host(&mut g, 0.3, 0.2, 2, 1);
    assert_abs_diff_eq!(g[1], 0.3, epsilon = 1e-12);
    assert_abs_diff_eq!(g[2], 0.29, epsilon = 1e-12);
}
