//! The spinor representation above the hand-transcribed `l = 4`.
//!
//! # What this used to be
//!
//! `spinor_l_max_panic_defect.rs` — an `#[ignore]`d reproduction of the fact
//! that a spinor shell at `l = 5` made `cart_to_spinor_sf_2d` **panic**
//! ("l=5 > 4 not supported"), reachable from `eval_raw`, while the single-block
//! ket path returned empty coefficient rows and so handed back zeros with an
//! `Ok`. Both came from one place: the Clebsch-Gordan tables in
//! `c2spinor_coeffs.rs` were transcribed by hand for `l = 0..=4` and stopped.
//!
//! # What changed
//!
//! That is not a ceiling libcint has. Its `g_c2s[]` carries spinor coefficients
//! for `l = 0..=12`, so the table is now *generated* from the vendored source
//! (`xtask gen-c2spinor-table`, drift-gated with `--check`), the accessors read
//! it, `SPINOR_L_MAX = 12` is enforced at `Shell::try_new` like `SPHERIC_L_MAX`,
//! and the transform's catch-all is a backstop nobody can reach through a public
//! entry point. The `l <= 4` blocks of the generated table are pinned to the
//! hand-transcribed ones bit for bit in `c2spinor::table_tests`.
//!
//! # What this gate is, and why it is built the way it is
//!
//! The first version compared `int1e_ovlp_spinor` end to end against libcint at
//! `(l, l)` and failed from `l = 9`. The probe that followed
//! (`cart`/`sph`/`spinor` side by side, same fixture) showed the residual was
//! not the fold's: the **Cartesian overlap block itself** drifted from libcint
//! as `l` grew, and the spherical and spinor folds of it carried a *smaller*
//! `diff / peak` than the Cartesian block at every `l`. (That drift was the 1e
//! VRR branch, since fixed — see `one_electron_adaptive_branch_parity` — but a
//! smaller residual of the same shape remains, from compiled-code multiply-add
//! contraction.) An end-to-end comparison at `(l, l)` therefore measures the 1e
//! recurrence, not the transform, and a gate that fails on the wrong component
//! is worse than none.
//!
//! So the transform is gated on its own terms:
//!
//! 1. [`spinor_fold_of_the_vendors_cartesian_block_matches_vendor`] takes
//!    libcint's **own** `int1e_ovlp_cart` / `int1e_kin_cart` block, pushes it
//!    through cintx's `cart_to_spinor_sf_2d`, and compares with libcint's
//!    `int1e_ovlp_spinor` / `int1e_kin_spinor`. Same input on both sides; only
//!    the fold differs. Agreement is held to `1e-12` of the block peak — the
//!    fold is a contraction with table constants, so anything beyond
//!    accumulation-order noise is a wrong coefficient.
//! 2. [`spinor_fold_does_not_amplify_the_kernels_own_residual`] is the
//!    end-to-end row, with the honest bound: cintx's spinor block may miss
//!    libcint's by no more (relative to peak) than cintx's Cartesian block
//!    already does. That Cartesian residual is printed by
//!    [`cartesian_residual_is_reported_for_the_fold_bound`]; the mechanism
//!    behind it is gated in `one_electron_adaptive_branch_parity`.
//! 3. [`spinor_fold_over_rys_blocks_matches_vendor`]: nuclear attraction and
//!    `int2e` end to end, at `(l, 0)` / `(l, 0, 0, 0)` so the recurrence depth
//!    stays on one side, for every `l` the build's device ceiling admits.
//! 4. `l = 13` is refused with a typed error — the fail-closed half.
//!
//! `kappa` is swept over `{-1, 0, +1}` in the fold gate, so the `j = l - 1/2`
//! block, the `j = l + 1/2` block, and the `kappa == 0` LT-then-GT over-read are
//! each exercised at every `l`.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_cubecl::backend::ResolvedBackend;
use cintx_cubecl::device_rys_ceiling::{RysFamily, device_nroots_ceiling};
use cintx_cubecl::transform::c2spinor::{cart_to_spinor_sf_2d, spinor_len};
use cintx_oracle::vendor_ffi;
use cintx_runtime::{BackendIntent, BackendKind};

/// Floors for the Rys-backed end-to-end rows, the same pair every extended-Rys
/// gate uses.
const ATOL: f64 = 1e-11;
const RTOL: f64 = 1e-9;

/// The fold alone, against the same Cartesian input: accumulation-order noise
/// and nothing else.
const FOLD_PEAK_RTOL: f64 = 1e-12;

/// The generated table's ceiling — libcint's own.
const LMAX: u8 = cintx_core::SPINOR_L_MAX;

fn ncart(l: u8) -> usize {
    let l = usize::from(l);
    (l + 1) * (l + 2) / 2
}

/// Two centres 2.2 bohr apart, plus an off-axis third nucleus so the nuclear
/// attraction sums over more than one centre. Shell `i` sits on centre `i % 2`
/// with angular momentum `ls[i]` and the given `kappa`.
fn fixture(ls: &[u8], kappa: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.6, 0.8, 0.5]];
    let charges = [8, 8, 1];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 3];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for index in 0..3 {
        atm[index * ATM_SLOTS + CHARGE_OF] = charges[index];
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = 0;
    }
    let mut bas = vec![0_i32; ls.len() * BAS_SLOTS];
    for (index, &l) in ls.iter().enumerate() {
        let exp_ptr = env.len() as i32;
        env.push(1.1 + 0.15 * index as f64);
        let coeff_ptr = env.len() as i32;
        env.push(1.0);
        bas[index * BAS_SLOTS + ATOM_OF] = (index % 2) as i32;
        bas[index * BAS_SLOTS + ANG_OF] = i32::from(l);
        bas[index * BAS_SLOTS + NPRIM_OF] = 1;
        bas[index * BAS_SLOTS + NCTR_OF] = 1;
        bas[index * BAS_SLOTS + KAPPA_OF] = kappa;
        bas[index * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[index * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn cpu() -> ResolvedBackend {
    ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        ..Default::default()
    })
    .expect("cpu backend")
}

fn peak(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn max_abs_diff(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(e, a)| (e - a).abs())
        .fold(0.0_f64, f64::max)
}

/// Overlap or kinetic: the two 1e symbols with no Rys quadrature.
#[derive(Clone, Copy)]
enum Scalar {
    Ovlp,
    Kin,
}

impl Scalar {
    fn name(self) -> &'static str {
        match self {
            Self::Ovlp => "ovlp",
            Self::Kin => "kin",
        }
    }
    fn vendor_cart(self, out: &mut [f64], atm: &[i32], bas: &[i32], env: &[f64]) {
        match self {
            Self::Ovlp => vendor_ffi::vendor_int1e_ovlp_cart(out, &[0, 1], atm, 3, bas, 2, env),
            Self::Kin => vendor_ffi::vendor_int1e_kin_cart(out, &[0, 1], atm, 3, bas, 2, env),
        };
    }
    fn vendor_spinor(self, out: &mut [f64], atm: &[i32], bas: &[i32], env: &[f64]) {
        match self {
            Self::Ovlp => vendor_ffi::vendor_int1e_ovlp_spinor(out, &[0, 1], atm, 3, bas, 2, env),
            Self::Kin => vendor_ffi::vendor_int1e_kin_spinor(out, &[0, 1], atm, 3, bas, 2, env),
        };
    }
    fn cintx_symbol(self, representation: &str) -> RawApiId {
        match (self, representation) {
            (Self::Ovlp, "cart") => RawApiId::Symbol("int1e_ovlp_cart"),
            (Self::Ovlp, _) => RawApiId::Symbol("int1e_ovlp_spinor"),
            (Self::Kin, "cart") => RawApiId::Symbol("int1e_kin_cart"),
            (Self::Kin, _) => RawApiId::Symbol("int1e_kin_spinor"),
        }
    }
}

/// **The transform, isolated.** libcint's own Cartesian block, folded by cintx,
/// against libcint's own spinor block. Every `l = 5..=12`, every `kappa`.
#[test]
fn spinor_fold_of_the_vendors_cartesian_block_matches_vendor() {
    for l in 5..=LMAX {
        for kappa in [-1_i32, 0, 1] {
            for scalar in [Scalar::Ovlp, Scalar::Kin] {
                let (atm, bas, env) = fixture(&[l, l], kappa);
                let nc = ncart(l);
                let ns = spinor_len(l, kappa);

                let mut cart = vec![0.0_f64; nc * nc];
                scalar.vendor_cart(&mut cart, &atm, &bas, &env);
                let mut expected = vec![0.0_f64; ns * ns * 2];
                scalar.vendor_spinor(&mut expected, &atm, &bas, &env);

                let mut folded = vec![0.0_f64; ns * ns * 2];
                cart_to_spinor_sf_2d::<f64>(&mut folded, &cart, l, kappa as i16, l, kappa as i16)
                    .unwrap_or_else(|e| panic!("fold refused l={l} kappa={kappa}: {e}"));

                let scale = peak(&expected);
                let diff = max_abs_diff(&expected, &folded);
                println!(
                    "  {:<4} l=({l},{l}) kappa={kappa:+} spinor={ns} peak={scale:.3e} \
                     max|diff|={diff:.3e} diff/peak={:.3e}",
                    scalar.name(),
                    diff / scale.max(f64::MIN_POSITIVE)
                );
                assert!(
                    scale > 1e-8,
                    "{} l={l} kappa={kappa}: vendor block peak {scale:.3e} is too near zero \
                     for agreement to mean anything",
                    scalar.name()
                );
                assert!(
                    diff <= FOLD_PEAK_RTOL * scale,
                    "{} l={l} kappa={kappa}: the fold of libcint's own Cartesian block \
                     differs from libcint's spinor block by {diff:.3e} on a peak of {scale:.3e} \
                     — a coefficient, not an accumulation order",
                    scalar.name()
                );
            }
        }
    }
}

/// **The end-to-end row, with the honest bound.** cintx's spinor block may miss
/// libcint's by no more, relative to the block peak, than cintx's own Cartesian
/// block already does (with 2x slack for the fold's accumulation order). The
/// Cartesian residual is printed per `l` because it *is* the finding: the 1e
/// overlap/kinetic recurrence drifts from libcint as `l` grows, and that is a
/// kernel property this gate reports but does not own.
#[test]
fn spinor_fold_does_not_amplify_the_kernels_own_residual() {
    for l in 5..=LMAX {
        for scalar in [Scalar::Ovlp, Scalar::Kin] {
            let (atm, bas, env) = fixture(&[l, l], 0);
            let nc = ncart(l);
            let ns = spinor_len(l, 0);

            let mut cart_e = vec![0.0_f64; nc * nc];
            scalar.vendor_cart(&mut cart_e, &atm, &bas, &env);
            let mut cart_a = vec![0.0_f64; nc * nc];
            let mut sp_e = vec![0.0_f64; ns * ns * 2];
            scalar.vendor_spinor(&mut sp_e, &atm, &bas, &env);
            let mut sp_a = vec![0.0_f64; ns * ns * 2];
            // SAFETY: both buffers are sized from the same extents the vendor
            // wrote for the same shells.
            unsafe {
                eval_raw(
                    scalar.cintx_symbol("cart"),
                    Some(&mut cart_a),
                    None,
                    &[0, 1],
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("{} cart l={l}: {e}", scalar.name()));
                eval_raw(
                    scalar.cintx_symbol("spinor"),
                    Some(&mut sp_a),
                    None,
                    &[0, 1],
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("{} spinor l={l}: {e}", scalar.name()));
            }

            let cart_residual =
                max_abs_diff(&cart_e, &cart_a) / peak(&cart_e).max(f64::MIN_POSITIVE);
            let spinor_residual = max_abs_diff(&sp_e, &sp_a) / peak(&sp_e).max(f64::MIN_POSITIVE);
            println!(
                "  {:<4} l=({l},{l}) cart residual/peak={cart_residual:.3e}  \
                 spinor residual/peak={spinor_residual:.3e}",
                scalar.name()
            );
            assert!(
                spinor_residual <= 2.0 * cart_residual.max(FOLD_PEAK_RTOL),
                "{} l={l}: the spinor fold amplified the kernel's residual — cart \
                 {cart_residual:.3e}, spinor {spinor_residual:.3e} (relative to peak)",
                scalar.name()
            );
        }
    }
}

/// The Cartesian residual this file's bound is written against.
///
/// When this gate was first written, cintx's Cartesian 1e overlap/kinetic
/// blocks drifted from libcint as `l` grew — 1.8e-13 of block peak at `l = 5`
/// rising to 3.8e-10 at `l = 12` — because the 1e VRR was always built on the
/// bra where libcint builds it on the shell with the larger angular momentum.
/// That is fixed, and `one_electron_adaptive_branch_parity` owns it now.
///
/// What remains is not cintx's to remove: a Python f64 emulation of libcint's
/// exact operation sequence reproduces libcint bit for bit and still differs
/// from cintx, so the tail is multiply-add contraction in the compiled kernel.
/// This test prints the residual it leaves, because
/// [`spinor_fold_does_not_amplify_the_kernels_own_residual`] is written relative
/// to it and a reader should be able to see the number that bound is made of.
#[test]
fn cartesian_residual_is_reported_for_the_fold_bound() {
    let mut worst = 0.0_f64;
    for l in 5..=LMAX {
        let (atm, bas, env) = fixture(&[l, l], 0);
        let nc = ncart(l);
        let mut e = vec![0.0_f64; nc * nc];
        Scalar::Ovlp.vendor_cart(&mut e, &atm, &bas, &env);
        let mut a = vec![0.0_f64; nc * nc];
        // SAFETY: sized from the vendor's own Cartesian extent.
        unsafe {
            eval_raw(
                RawApiId::Symbol("int1e_ovlp_cart"),
                Some(&mut a),
                None,
                &[0, 1],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        }
        .unwrap();
        let residual = max_abs_diff(&e, &a) / peak(&e).max(f64::MIN_POSITIVE);
        println!("  ovlp cart l=({l},{l}) residual/peak={residual:.3e}");
        worst = worst.max(residual);
    }
    // Loose on purpose: the point is to print the number, and to notice if the
    // recurrence ever regresses by orders of magnitude. The tight, mechanism-
    // testing gate is `one_electron_adaptive_branch_parity`.
    assert!(
        worst <= 1e-8,
        "the Cartesian 1e residual reached {worst:.3e} of block peak; the fold bound in          this file is written against a much smaller number"
    );
}

/// **The transform over a Rys block.** Nuclear attraction and `int2e`, end to
/// end, at `(l, 0)` / `(l, 0, 0, 0)` — the recurrence depth stays on one side,
/// so the Cartesian block is inside tolerance and the row measures the fold —
/// for every `l` whose Rys order this build's device ceiling admits.
#[test]
fn spinor_fold_over_rys_blocks_matches_vendor() {
    let backend = cpu();
    let ceiling_1e = device_nroots_ceiling(&backend, RysFamily::Int1e);
    let ceiling_2e = device_nroots_ceiling(&backend, RysFamily::Int2e);

    fn check(label: &str, ls: &[u8], expected: &[f64], actual: &[f64]) -> f64 {
        let (mut mismatches, mut worst) = (0_usize, 0.0_f64);
        for (index, (e, a)) in expected.iter().zip(actual).enumerate() {
            let diff = (e - a).abs();
            let tol = ATOL.max(RTOL * e.abs());
            worst = worst.max(diff / tol);
            if diff > tol {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  MISMATCH {label} l={ls:?} [{index}]: vendor={e:.15e} cintx={a:.15e}"
                    );
                }
            }
        }
        assert!(
            peak(expected) > 1e-14,
            "{label} l={ls:?}: vendor block is all zero"
        );
        assert_eq!(
            mismatches,
            0,
            "{label} l={ls:?}: {mismatches} of {} mismatched",
            expected.len()
        );
        worst
    }

    let mut covered = Vec::new();
    for l in 5..=LMAX {
        let order = usize::from(l) / 2 + 1;
        if order <= ceiling_1e {
            let ls = [l, 0];
            let (atm, bas, env) = fixture(&ls, 0);
            let n = spinor_len(l, 0) * spinor_len(0, 0) * 2;
            let mut e = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_nuc_spinor(&mut e, &[0, 1], &atm, 3, &bas, 2, &env);
            let mut a = vec![0.0_f64; n];
            // SAFETY: sized from the vendor's spinor extents for these shells.
            unsafe {
                eval_raw(
                    RawApiId::Symbol("int1e_nuc_spinor"),
                    Some(&mut a),
                    None,
                    &[0, 1],
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
            }
            .unwrap_or_else(|err| panic!("int1e_nuc_spinor l={l}: {err}"));
            let worst = check("int1e_nuc_spinor", &ls, &e, &a);
            println!(
                "  int1e_nuc_spinor  l=({l},0)      nroots={order} elements={n} worst |diff|/tol={worst:.3e}"
            );
        }
        if order <= ceiling_2e {
            let ls = [l, 0, 0, 0];
            let (atm, bas, env) = fixture(&ls, 0);
            let n = spinor_len(l, 0) * spinor_len(0, 0).pow(3) * 2;
            let mut e = vec![0.0_f64; n];
            vendor_ffi::vendor_int2e_spinor(&mut e, &[0, 1, 2, 3], &atm, 3, &bas, 4, &env);
            let mut a = vec![0.0_f64; n];
            // SAFETY: as above, for the quartet.
            unsafe {
                eval_raw(
                    RawApiId::Symbol("int2e_spinor"),
                    Some(&mut a),
                    None,
                    &[0, 1, 2, 3],
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
            }
            .unwrap_or_else(|err| panic!("int2e_spinor l={l}: {err}"));
            let worst = check("int2e_spinor", &ls, &e, &a);
            println!(
                "  int2e_spinor      l=({l},0,0,0)  nroots={order} elements={n} worst |diff|/tol={worst:.3e}"
            );
            covered.push(l);
        }
    }
    println!(
        "int2e_spinor covered l={covered:?} under device ceiling {ceiling_2e} (extended-device-rys {})",
        if cintx_cubecl::EXTENDED_DEVICE_RYS_COMPILED {
            "on"
        } else {
            "off"
        }
    );
    assert!(
        !covered.is_empty(),
        "no order was admitted; the ceiling is broken"
    );
}

/// **Fail-closed above the table.** `l = 13` has no coefficients in libcint
/// either, so there is no reference to be compatible with and the right answer
/// is a typed refusal — the panic this file was written to record is gone, and
/// so is the silent-zero alternative.
#[test]
fn spinor_above_the_table_ceiling_refuses_instead_of_panicking() {
    let (atm, bas, env) = fixture(&[LMAX + 1, LMAX + 1], 0);
    let n = spinor_len(LMAX + 1, 0);
    let mut out = vec![0.0_f64; n * n * 2];
    // SAFETY: the buffer is sized for the full block; the call is expected to
    // refuse before writing anything.
    let status = unsafe {
        eval_raw(
            RawApiId::Symbol("int1e_ovlp_spinor"),
            Some(&mut out),
            None,
            &[0, 1],
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    };
    let err = status.expect_err("l past SPINOR_L_MAX must be a typed refusal");
    let text = err.to_string();
    assert!(
        text.contains("SPINOR_L_MAX"),
        "the refusal should name the ceiling it applied: {text}"
    );
    assert!(
        out.iter().all(|v| *v == 0.0),
        "a refused call must not have written into the caller's buffer"
    );
}
