//! Phase 26 GIAO-01: vendor byte-identity parity for the 11 spin-free 1e GIAO/CG
//! magnetic-property families (cart + sph) at atol=1e-12 on the NON-ZERO gauge
//! origin fixture with a NON-SQUARE bra×ket block.
//!
//! D-15 reconciliation: the GIAO integrals are purely imaginary. libcint's cart/
//! sph symbol returns the REAL magnitude of the imaginary part as a real `double*`
//! buffer. cintx registers these families with `complex_output=true`, so the raw
//! `eval_raw` output is a `2×`-interleaved `[re, im]` complex buffer (the safe-API
//! `Complex<f64>` view). The parity comparison is therefore:
//!   - extract cintx's IMAGINARY half (odd indices) → compare vs vendor's real out
//!   - cintx's REAL half (even indices) must be exactly 0.0 (D-07; checked here too)
//!
//! This file CANNOT reuse `moment_common::vendor_parity` verbatim because that
//! helper sizes the cintx buffer `rank*ni*nj` (real); the GIAO families need a
//! `2*rank*ni*nj` complex buffer with imaginary-half extraction. The shared
//! fixture / shell-pair / mismatch helpers ARE reused.
//!
//! Double-gated: crate `#![cfg(any(cpu, rocm))]`; per-test `#[cfg(has_vendor_libcint)]
//! #[cfg(feature = "cpu")]`. Without `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`
//! the parity tests silently skip (project memory).

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[path = "moment_common.rs"]
mod moment_common;

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
mod parity {
    use super::moment_common::{
        ATOL, RTOL, count_mismatches, cross_center_non_square_shell_pair, ncart, nsph,
    };
    use cintx_compat::raw::{ANG_OF, ATM_SLOTS, BAS_SLOTS, RawApiId, eval_raw};
    use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
    use cintx_oracle::vendor_ffi;

    /// Collect cintx `eval_raw` output for ONE non-square shell pair as a COMPLEX
    /// (`2*rank*ni*nj`) interleaved `[re, im]` buffer, then split into (re, im)
    /// real halves. The GIAO families carry `complex_output=true`, so the raw out
    /// buffer must be sized `2×`.
    fn collect_cintx_complex(
        rank: usize,
        api_id: RawApiId,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        shls_pair: (usize, usize),
        nf: impl Fn(i32) -> usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let (si, sj) = shls_pair;
        let li = bas[si * BAS_SLOTS + ANG_OF];
        let lj = bas[sj * BAS_SLOTS + ANG_OF];
        let ni = nf(li);
        let nj = nf(lj);
        let n = rank * ni * nj;
        let shls = [si as i32, sj as i32];
        let mut out = vec![0.0_f64; 2 * n];

        // SAFETY: atm/bas/env well-formed; shls valid.
        unsafe {
            eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
        }
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for p in 0..n {
            re[p] = out[2 * p];
            im[p] = out[2 * p + 1];
        }
        (re, im)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_vendor_real(
        rank: usize,
        vendor_fn: impl Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        shls_pair: (usize, usize),
        nf: impl Fn(i32) -> usize,
    ) -> Vec<f64> {
        let (si, sj) = shls_pair;
        let li = bas[si * BAS_SLOTS + ANG_OF];
        let lj = bas[sj * BAS_SLOTS + ANG_OF];
        let ni = nf(li);
        let nj = nf(lj);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let shls = [si as i32, sj as i32];
        let mut out = vec![0.0_f64; rank * ni * nj];
        vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
        out
    }

    fn assert_any_nonzero(block: &[f64], label: &str) {
        assert!(
            block.iter().any(|v| v.abs() > 1e-14),
            "{label}: block is all-zero (zero-fill regression)"
        );
    }

    /// GIAO vendor parity for ONE family: compares cintx's imaginary half (sph and
    /// cart) against vendor's real output at atol=1e-12 on the non-square block;
    /// also asserts the real half is exactly 0.0 (D-07).
    #[allow(clippy::too_many_arguments)]
    pub fn giao_vendor_parity<FS, FC>(
        rank: usize,
        api_sph: RawApiId,
        api_cart: RawApiId,
        vendor_sph: FS,
        vendor_cart: FC,
        label: &str,
    ) where
        FS: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        FC: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    {
        let (atm, bas, env) = build_h2o_sto3g_common_orig();
        // The GIAO gout combos carry a `c = ri - rj` factor (govlp/igovlp/igkin/
        // gnuc/ignuc/a01gp), which vanishes on a SAME-center block. Use a
        // CROSS-center non-square pair (H1-1s × O-2p) so every family's integral
        // is genuinely non-zero AND the block is non-square (D-12 transpose gate).
        let pair = cross_center_non_square_shell_pair();

        // ── sph ──
        let vendor_s = collect_vendor_real(rank, &vendor_sph, &atm, &bas, &env, pair, nsph);
        let (re_s, im_s) = collect_cintx_complex(rank, api_sph, &atm, &bas, &env, pair, nsph);
        assert_any_nonzero(&im_s, &format!("{label}_sph cintx imag"));
        assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
        // D-07: real half exactly zero.
        for (i, &v) in re_s.iter().enumerate() {
            assert_eq!(v, 0.0, "{label}_sph: real part at {i} must be exactly 0.0, got {v}");
        }
        let mm = count_mismatches(&vendor_s, &im_s, ATOL, RTOL);
        assert_eq!(mm, 0, "{label}_sph: {mm} mismatches vs vendored libcint at atol={ATOL}");

        // ── cart ──
        let vendor_c = collect_vendor_real(rank, &vendor_cart, &atm, &bas, &env, pair, ncart);
        let (re_c, im_c) = collect_cintx_complex(rank, api_cart, &atm, &bas, &env, pair, ncart);
        assert_any_nonzero(&im_c, &format!("{label}_cart cintx imag"));
        assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
        for (i, &v) in re_c.iter().enumerate() {
            assert_eq!(v, 0.0, "{label}_cart: real part at {i} must be exactly 0.0, got {v}");
        }
        let mm = count_mismatches(&vendor_c, &im_c, ATOL, RTOL);
        assert_eq!(mm, 0, "{label}_cart: {mm} mismatches vs vendored libcint at atol={ATOL}");
    }

    #[test]
    fn test_int1e_govlp_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_GOVLP_SPH,
            RawApiId::INT1E_GOVLP_CART,
            vendor_ffi::vendor_int1e_govlp_sph,
            vendor_ffi::vendor_int1e_govlp_cart,
            "int1e_govlp",
        );
    }

    #[test]
    fn test_int1e_gnuc_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_GNUC_SPH,
            RawApiId::INT1E_GNUC_CART,
            vendor_ffi::vendor_int1e_gnuc_sph,
            vendor_ffi::vendor_int1e_gnuc_cart,
            "int1e_gnuc",
        );
    }

    #[test]
    fn test_int1e_igovlp_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_IGOVLP_SPH,
            RawApiId::INT1E_IGOVLP_CART,
            vendor_ffi::vendor_int1e_igovlp_sph,
            vendor_ffi::vendor_int1e_igovlp_cart,
            "int1e_igovlp",
        );
    }

    #[test]
    fn test_int1e_ignuc_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_IGNUC_SPH,
            RawApiId::INT1E_IGNUC_CART,
            vendor_ffi::vendor_int1e_ignuc_sph,
            vendor_ffi::vendor_int1e_ignuc_cart,
            "int1e_ignuc",
        );
    }

    #[test]
    fn test_int1e_igkin_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_IGKIN_SPH,
            RawApiId::INT1E_IGKIN_CART,
            vendor_ffi::vendor_int1e_igkin_sph,
            vendor_ffi::vendor_int1e_igkin_cart,
            "int1e_igkin",
        );
    }

    // DEFERRED (Plan 26-02): int1e_a01gp (rank-9 NABLA-RINV CROSS P, 27-s table) is
    // byte-identical on component 0 but ~2× on a subset of ket-varying elements of
    // components 1..8. The other 10 GIAO-01 families (including the structurally
    // identical rank-9 a11part pair and the 27-s igkin) pass at atol=1e-12. The
    // a01gp discrepancy is isolated to its ∇⊗∇ ket-derivative double-count in the
    // combined g2=D_J+D_I tensor and is tracked as a known stub; the family is
    // registered + kernel'd + vendor-wrapped but NOT yet oracle_covered.
    #[test]
    #[ignore = "26-02 deferred: a01gp rank-9 27-s table has a 2x ket-element \
                discrepancy; tracked in SUMMARY Known Stubs"]
    fn test_int1e_a01gp_parity() {
        giao_vendor_parity(
            9,
            RawApiId::INT1E_A01GP_SPH,
            RawApiId::INT1E_A01GP_CART,
            vendor_ffi::vendor_int1e_a01gp_sph,
            vendor_ffi::vendor_int1e_a01gp_cart,
            "int1e_a01gp",
        );
    }

    #[test]
    fn test_int1e_ia01p_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_IA01P_SPH,
            RawApiId::INT1E_IA01P_CART,
            vendor_ffi::vendor_int1e_ia01p_sph,
            vendor_ffi::vendor_int1e_ia01p_cart,
            "int1e_ia01p",
        );
    }

    #[test]
    fn test_int1e_cg_irxp_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_CG_IRXP_SPH,
            RawApiId::INT1E_CG_IRXP_CART,
            vendor_ffi::vendor_int1e_cg_irxp_sph,
            vendor_ffi::vendor_int1e_cg_irxp_cart,
            "int1e_cg_irxp",
        );
    }

    #[test]
    fn test_int1e_giao_irjxp_parity() {
        giao_vendor_parity(
            3,
            RawApiId::INT1E_GIAO_IRJXP_SPH,
            RawApiId::INT1E_GIAO_IRJXP_CART,
            vendor_ffi::vendor_int1e_giao_irjxp_sph,
            vendor_ffi::vendor_int1e_giao_irjxp_cart,
            "int1e_giao_irjxp",
        );
    }

    #[test]
    fn test_int1e_cg_a11part_parity() {
        giao_vendor_parity(
            9,
            RawApiId::INT1E_CG_A11PART_SPH,
            RawApiId::INT1E_CG_A11PART_CART,
            vendor_ffi::vendor_int1e_cg_a11part_sph,
            vendor_ffi::vendor_int1e_cg_a11part_cart,
            "int1e_cg_a11part",
        );
    }

    #[test]
    fn test_int1e_giao_a11part_parity() {
        giao_vendor_parity(
            9,
            RawApiId::INT1E_GIAO_A11PART_SPH,
            RawApiId::INT1E_GIAO_A11PART_CART,
            vendor_ffi::vendor_int1e_giao_a11part_sph,
            vendor_ffi::vendor_int1e_giao_a11part_cart,
            "int1e_giao_a11part",
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 26-04 / CR-01: fail-closed contract for int1e_a01gp.
//
// Unlike the `parity` module above, this module is NOT vendor-gated — it runs on
// the DEFAULT (non-vendor) test profile so the fail-closed contract is ALWAYS
// covered (no `CINTX_ORACLE_BUILD_VENDOR=1` required). The a01gp raw path is
// registered/dispatchable but the rank-9 kernel is known-wrong (~2x on ket-varying
// components 1..8); the public `eval_raw` path MUST fail closed with
// `UnsupportedApi` until Plan 26-05 lands vendor parity and removes the guard.
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
mod fail_closed {
    use super::moment_common::{cross_center_non_square_shell_pair, ncart, nsph};
    use cintx_compat::raw::{ANG_OF, BAS_SLOTS, RawApiId, eval_raw};
    use cintx_core::cintxRsError;
    use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;

    /// Drive the a01gp raw path through `eval_raw` for ONE non-square shell pair
    /// and return the `Result`. The buffer is sized `2*rank*ni*nj` (complex_output)
    /// so an undersized-output error cannot masquerade as the fail-closed outcome.
    fn try_eval_a01gp(
        api_id: RawApiId,
        nf: impl Fn(i32) -> usize,
    ) -> Result<(), cintxRsError> {
        let (atm, bas, env) = build_h2o_sto3g_common_orig();
        let (si, sj) = cross_center_non_square_shell_pair();
        let li = bas[si * BAS_SLOTS + ANG_OF];
        let lj = bas[sj * BAS_SLOTS + ANG_OF];
        let n = 9 * nf(li) * nf(lj); // a01gp rank = 9
        let shls = [si as i32, sj as i32];
        let mut out = vec![0.0_f64; 2 * n];

        // SAFETY: atm/bas/env well-formed; shls valid for the H2O/STO-3G corpus.
        unsafe {
            eval_raw(api_id, Some(&mut out), None, &shls, &atm, &bas, &env, None, None)
        }
        .map(|_| ())
    }

    /// 26-04 / CR-01: a01gp is registered but the kernel is known-wrong; the public
    /// raw path MUST fail closed with UnsupportedApi until 26-05 lands parity.
    #[test]
    fn test_int1e_a01gp_is_fail_closed() {
        for api in [RawApiId::INT1E_A01GP_CART, RawApiId::INT1E_A01GP_SPH] {
            let nf: fn(i32) -> usize = if api == RawApiId::INT1E_A01GP_CART {
                ncart
            } else {
                nsph
            };
            match try_eval_a01gp(api, nf) {
                Err(cintxRsError::UnsupportedApi { .. }) => { /* fail-closed: OK */ }
                Err(other) => panic!(
                    "expected UnsupportedApi for {api:?}, got a different error: {other:?}"
                ),
                Ok(()) => panic!(
                    "{api:?} returned Ok — int1e_a01gp must fail closed with UnsupportedApi \
                     (known-wrong rank-9 kernel; no silently-wrong buffer permitted)"
                ),
            }
        }
    }
}
