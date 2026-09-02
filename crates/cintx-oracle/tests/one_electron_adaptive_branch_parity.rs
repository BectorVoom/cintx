//! The 1e VRR is built on the shell with the larger angular momentum, as
//! libcint's is.
//!
//! # The defect this pins
//!
//! `CINTinit_int1e_EnvVars` (g1e.c) splits on `ibase = li_ceil > lj_ceil`, and
//! `CINTg1e_ovlp` / `CINTg1e_nuc` build the vertical recurrence on whichever
//! shell carries the larger *ceiling* angular momentum, transferring to the
//! smaller one. cintx's scalar 1e kernel always built on the bra — its host
//! reference said so in as many words, "For simplicity we always VRR on bra
//! (center i) then HRR to ket".
//!
//! That is the same answer in exact arithmetic and a differently-rounded one in
//! f64, and the difference grows with `l` because the HRR's
//! `G(i, j) = G(i+1, j-1) + rirj * G(i, j-1)` subtracts when `rirj` is negative.
//! Verified against a 60-digit mpmath reference: on an `(l, l)` Cartesian
//! overlap **both** engines drift from exact — libcint by 1.8e-10 of the block
//! peak at `l = 12` — so the goal was never to beat libcint, only to round the
//! way it does. That is what result compatibility means here.
//!
//! # Why the gate is an *asymmetry* bound
//!
//! An absolute bound would have to be loose enough to admit the part of the
//! disagreement that is not cintx's to fix: a Python f64 emulation of libcint's
//! exact operation sequence reproduces libcint **bit for bit** at every `l`, and
//! still differs from cintx — the remainder is floating-point contraction in the
//! compiled kernel (the CubeCL backend fuses multiply-adds that libcint's C
//! build does not), which is codegen, not algorithm, and differs per backend.
//!
//! What *is* cintx's to fix shows up as a direction: with the branch missing,
//! the error depended on which side carried the larger `l`. `(12, 5)` agreed
//! with libcint to 1.1e-12 while its transpose `(5, 12)` was 1.4e-10 out — the
//! same integral, 127x worse for being written the other way round. Bounding
//! that ratio tests exactly the thing the branch controls and is insensitive to
//! the contraction noise floor.
//!
//! Measured over the full `li, lj = 0..=12` grid on the fixture below:
//!
//! | | max orientation asymmetry | pairs over 30x |
//! |---|---|---|
//! | overlap, before | 833x | 35 / 78 |
//! | overlap, after | **19.4x** | **0 / 78** |
//! | kinetic, before | 3670x | 41 / 78 |
//! | kinetic, after | **17.5x** | **0 / 78** |

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;

/// Highest `l` swept. libcint's Cartesian tables run further, but `l = 12` is
/// where the spinor coupling table stops and is already well past any published
/// orbital basis.
const LMAX: u8 = 12;

/// No orientation pair may disagree with libcint by more than this factor of
/// its transpose.
///
/// The post-fix maximum is 19.4x (overlap) and 17.5x (kinetic); the pre-fix
/// maxima were 833x and 3670x. 30x sits clear of the former and far below the
/// latter, so this catches the branch going missing without tracking the
/// backend's contraction noise.
const MAX_ORIENTATION_ASYMMETRY: f64 = 30.0;

/// Floor below which a ratio is meaningless: two blocks that both agree to a
/// few ulp are not "asymmetric" because one happens to be exact.
const RATIO_FLOOR: f64 = 1e-15;

/// Absolute ceilings, from the measured worst over the whole grid (overlap
/// 1.54e-11, kinetic 3.58e-10) with roughly 3x headroom. The pre-fix worsts were
/// 5.64e-11 and 5.87e-9, so these discriminate too — but the asymmetry gate
/// above is the one that tests the mechanism.
const MAX_OVLP_RESIDUAL: f64 = 5e-11;
const MAX_KIN_RESIDUAL: f64 = 1.5e-9;

/// Two shells on centres in **general position** — neither at the origin, and
/// off-axis.
///
/// This matters. An earlier fixture put the bra at the origin, which collapses
/// `rij = ri + wj * (rj - ri)` and `(ai * ri + aj * rj) / aij` onto the same
/// expression and so cannot tell libcint's association from any other. It also
/// made the `li == lj` classes behave unrepresentatively.
fn fixture(li: u8, lj: u8) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [[0.31, -0.22, 0.17], [0.44, 0.91, 2.05], [0.6, 0.8, 0.5]];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 3];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for index in 0..3 {
        atm[index * ATM_SLOTS + CHARGE_OF] = 8;
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    }
    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for (index, &l) in [li, lj].iter().enumerate() {
        let exp_ptr = env.len() as i32;
        env.push(1.1 + 0.15 * index as f64);
        let coeff_ptr = env.len() as i32;
        env.push(1.0);
        bas[index * BAS_SLOTS + ATOM_OF] = index as i32;
        bas[index * BAS_SLOTS + ANG_OF] = i32::from(l);
        bas[index * BAS_SLOTS + NPRIM_OF] = 1;
        bas[index * BAS_SLOTS + NCTR_OF] = 1;
        bas[index * BAS_SLOTS + KAPPA_OF] = 0;
        bas[index * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[index * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn ncart(l: u8) -> usize {
    let l = usize::from(l);
    (l + 1) * (l + 2) / 2
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Ovlp,
    Kin,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Ovlp => "int1e_ovlp_cart",
            Self::Kin => "int1e_kin_cart",
        }
    }
    fn ceiling(self) -> f64 {
        match self {
            Self::Ovlp => MAX_OVLP_RESIDUAL,
            Self::Kin => MAX_KIN_RESIDUAL,
        }
    }
}

/// `max |cintx - libcint| / peak(libcint)` for one class, or `None` when the
/// block is numerically zero (nothing to compare) or the call is refused.
fn residual(op: Op, li: u8, lj: u8) -> Option<f64> {
    let (atm, bas, env) = fixture(li, lj);
    let n = ncart(li) * ncart(lj);
    let mut expected = vec![0.0_f64; n];
    match op {
        Op::Ovlp => {
            vendor_ffi::vendor_int1e_ovlp_cart(&mut expected, &[0, 1], &atm, 3, &bas, 2, &env)
        }
        Op::Kin => {
            vendor_ffi::vendor_int1e_kin_cart(&mut expected, &[0, 1], &atm, 3, &bas, 2, &env)
        }
    };
    let mut actual = vec![0.0_f64; n];
    // SAFETY: `actual` is sized from the same Cartesian AO counts the vendor
    // writes for these shells.
    unsafe {
        eval_raw(
            RawApiId::Symbol(op.name()),
            Some(&mut actual),
            None,
            &[0, 1],
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .ok()?;
    let peak = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if peak <= 1e-30 {
        return None;
    }
    let worst = expected
        .iter()
        .zip(&actual)
        .map(|(e, a)| (e - a).abs())
        .fold(0.0_f64, f64::max);
    Some(worst / peak)
}

/// **The gate.** Agreement with libcint must not depend on which shell carries
/// the larger angular momentum.
#[test]
fn the_vrr_branch_follows_the_larger_angular_momentum() {
    for op in [Op::Ovlp, Op::Kin] {
        let mut worst_ratio = 0.0_f64;
        let mut worst_pair = (0_u8, 0_u8);
        let mut offenders = Vec::new();
        let mut compared = 0_usize;

        for li in 0..=LMAX {
            for lj in (li + 1)..=LMAX {
                let (Some(a), Some(b)) = (residual(op, li, lj), residual(op, lj, li)) else {
                    continue;
                };
                compared += 1;
                let ratio = a.max(b).max(RATIO_FLOOR) / a.min(b).max(RATIO_FLOOR);
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_pair = (li, lj);
                }
                if ratio > MAX_ORIENTATION_ASYMMETRY {
                    offenders.push(format!(
                        "({li},{lj})={a:.2e} vs ({lj},{li})={b:.2e} -> {ratio:.0}x"
                    ));
                }
            }
        }

        println!(
            "  {}: {compared} orientation pairs, worst asymmetry {worst_ratio:.1}x at \
             (li,lj)={:?}",
            op.name(),
            worst_pair
        );
        assert!(
            compared > 50,
            "{}: only {compared} pairs compared",
            op.name()
        );
        assert!(
            offenders.is_empty(),
            "{}: {} orientation pair(s) disagree with libcint by more than \
             {MAX_ORIENTATION_ASYMMETRY}x depending on which side carries the larger l — \
             the adaptive VRR branch is not being applied: {:?}",
            op.name(),
            offenders.len(),
            &offenders[..offenders.len().min(8)]
        );
    }
}

/// The absolute envelope, for the record. Not a claim that cintx is exact —
/// libcint is itself ~1.8e-10 of block peak from a 60-digit reference at
/// `l = 12`, and the last part of the disagreement is compiled-code
/// multiply-add contraction rather than anything algorithmic.
#[test]
fn high_l_cartesian_blocks_track_libcint() {
    for op in [Op::Ovlp, Op::Kin] {
        let mut worst = 0.0_f64;
        let mut worst_pair = (0_u8, 0_u8);
        for li in 0..=LMAX {
            for lj in 0..=LMAX {
                let Some(r) = residual(op, li, lj) else {
                    continue;
                };
                if r > worst {
                    worst = r;
                    worst_pair = (li, lj);
                }
            }
        }
        println!(
            "  {}: worst residual/peak {worst:.3e} at (li,lj)={worst_pair:?} \
             (ceiling {:.1e})",
            op.name(),
            op.ceiling()
        );
        assert!(
            worst <= op.ceiling(),
            "{}: worst |cintx - libcint| / peak is {worst:.3e} at {worst_pair:?}, over the \
             {:.1e} ceiling",
            op.name(),
            op.ceiling()
        );
    }
}
