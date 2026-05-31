//! Phase 26 GIAO-02 (D-16): vendor byte-identity parity for the 4 spin-free 2e
//! GIAO magnetic-property families (cart + sph) at atol=1e-12 on the NON-ZERO
//! gauge-origin fixture with a NON-SQUARE cross-center bra×ket quartet.
//!
//! Families: int2e_g1 (rank 3), int2e_ig1 (rank 3), int2e_gg1 (rank 9),
//! int2e_g1g2 (rank 9; IN scope per D-16 — registered, kernel-ported, and
//! parity-tested here alongside the other three).
//!
//! D-15 reconciliation: the GIAO integrals are purely imaginary. libcint's cart/
//! sph symbol returns the REAL magnitude of the imaginary part as a real `double*`
//! buffer. cintx registers these families with `complex_output=true`, so the raw
//! `eval_raw` output is a `2×`-interleaved `[re, im]` complex buffer. The parity
//! comparison is therefore:
//!   - extract cintx's IMAGINARY half (odd indices) → compare vs vendor's real out
//!   - cintx's REAL half (even indices) must be exactly 0.0 (D-07; checked here too)
//!
//! NON-SQUARE CROSS-CENTER QUARTET (D-12): the 2e GIAO gout carries a `c = ri-rj`
//! factor (g1/ig1/gg1) — and `c = (ri-rj)⊗(rk-rl)` for g1g2 — which vanishes on a
//! same-center bra (or ket). The quartet [H1-1s, O-2p, H1-1s, O-2p] = [3,2,3,2] is
//! cross-center on BOTH electron pairs (rirj≠0 AND rkrl≠0) and non-square (l=0 vs
//! l=1), so a transposed layout cannot pass and g1g2's both-electron factor is
//! genuinely exercised.
//!
//! Double-gated: crate `#![cfg(any(cpu, rocm))]`; per-test `#[cfg(has_vendor_libcint)]
//! #[cfg(feature = "cpu")]`. Without `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`
//! the parity tests silently skip (project memory). The folded-todo
//! `oracle-cart-offset-vendor-zero` (`compare::tests` `--lib` failures) is a
//! confirmed pre-existing harness bug — family parity here is `--test` integration.

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[path = "moment_common.rs"]
mod moment_common;

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
mod parity {
    // Single source of truth for parity tolerances + mismatch logic (WR-03): the
    // tolerance/mismatch/zero-fill helpers are shared with `giao_1e_parity.rs` and the
    // Phase 24 moment parity scaffolds via `moment_common`, so a tolerance change tracks
    // every parity file rather than silently drifting. moment_common's
    // `count_mismatches(ref, obs, atol, rtol)` takes the tolerances as arguments (vs the
    // former local 2-arg copy that hard-coded ATOL/RTOL inside the fn); call sites pass
    // `ATOL, RTOL` so the gate semantics are byte-identical (atol=1e-12, rtol=0).
    use super::moment_common::{ATOL, RTOL, assert_any_nonzero, count_mismatches, ncart, nsph};
    use cintx_compat::raw::{ANG_OF, ATM_SLOTS, BAS_SLOTS, RawApiId, eval_raw};
    use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
    use cintx_oracle::vendor_ffi;

    /// A NON-SQUARE quartet on cross-center bra AND cross-center ket electron pairs.
    /// H2O/STO-3G shells: 0=O-1s, 1=O-2s, 2=O-2p(l=1), 3=H1-1s(l=0), 4=H2-1s(l=0).
    /// [H1-1s, O-2p, H1-1s, O-2p] → rirj≠0 (e1) AND rkrl≠0 (e2), non-square (D-12).
    fn cross_center_non_square_quartet() -> [usize; 4] {
        [3, 2, 3, 2]
    }

    /// Collect cintx `eval_raw` output for ONE non-square quartet as a COMPLEX
    /// (`2*rank*ni*nj*nk*nl`) interleaved `[re, im]` buffer, split into (re, im).
    fn collect_cintx_complex(
        rank: usize,
        api_id: RawApiId,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        quartet: [usize; 4],
        nf: impl Fn(i32) -> usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let l: Vec<i32> = quartet.iter().map(|&s| bas[s * BAS_SLOTS + ANG_OF]).collect();
        let nblock: usize = l.iter().map(|&li| nf(li)).product();
        let n = rank * nblock;
        let shls = [
            quartet[0] as i32,
            quartet[1] as i32,
            quartet[2] as i32,
            quartet[3] as i32,
        ];
        let mut out = vec![0.0_f64; 2 * n];
        // SAFETY: atm/bas/env well-formed; shls valid.
        unsafe {
            eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                .unwrap_or_else(|e| panic!("eval_raw failed for quartet {quartet:?}: {e:?}"));
        }
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for p in 0..n {
            re[p] = out[2 * p];
            im[p] = out[2 * p + 1];
        }
        (re, im)
    }

    fn collect_vendor_real(
        rank: usize,
        vendor_fn: impl Fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        quartet: [usize; 4],
        nf: impl Fn(i32) -> usize,
    ) -> Vec<f64> {
        let l: Vec<i32> = quartet.iter().map(|&s| bas[s * BAS_SLOTS + ANG_OF]).collect();
        let nblock: usize = l.iter().map(|&li| nf(li)).product();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let shls = [
            quartet[0] as i32,
            quartet[1] as i32,
            quartet[2] as i32,
            quartet[3] as i32,
        ];
        let mut out = vec![0.0_f64; rank * nblock];
        vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
        out
    }

    /// GIAO-2e vendor parity for ONE family: compares cintx's imaginary half (sph
    /// and cart) against vendor's real output at atol=1e-12 on the non-square
    /// cross-center quartet; also asserts the real half is exactly 0.0 (D-07).
    #[allow(clippy::too_many_arguments)]
    fn giao_2e_vendor_parity<FS, FC>(
        rank: usize,
        api_sph: RawApiId,
        api_cart: RawApiId,
        vendor_sph: FS,
        vendor_cart: FC,
        label: &str,
    ) where
        FS: Fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        FC: Fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    {
        let (atm, bas, env) = build_h2o_sto3g_common_orig();
        let quartet = cross_center_non_square_quartet();

        // ── sph ──
        let vendor_s = collect_vendor_real(rank, &vendor_sph, &atm, &bas, &env, quartet, nsph);
        let (re_s, im_s) = collect_cintx_complex(rank, api_sph, &atm, &bas, &env, quartet, nsph);
        assert_any_nonzero(&im_s, &format!("{label}_sph cintx imag"));
        assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
        for (i, &v) in re_s.iter().enumerate() {
            assert_eq!(v, 0.0, "{label}_sph: real part at {i} must be exactly 0.0, got {v}");
        }
        let mm = count_mismatches(&vendor_s, &im_s, ATOL, RTOL);
        assert_eq!(mm, 0, "{label}_sph: {mm} mismatches vs vendored libcint at atol={ATOL}");

        // ── cart ──
        let vendor_c = collect_vendor_real(rank, &vendor_cart, &atm, &bas, &env, quartet, ncart);
        let (re_c, im_c) = collect_cintx_complex(rank, api_cart, &atm, &bas, &env, quartet, ncart);
        assert_any_nonzero(&im_c, &format!("{label}_cart cintx imag"));
        assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
        for (i, &v) in re_c.iter().enumerate() {
            assert_eq!(v, 0.0, "{label}_cart: real part at {i} must be exactly 0.0, got {v}");
        }
        let mm = count_mismatches(&vendor_c, &im_c, ATOL, RTOL);
        assert_eq!(mm, 0, "{label}_cart: {mm} mismatches vs vendored libcint at atol={ATOL}");
    }

    #[test]
    fn test_int2e_g1_parity() {
        giao_2e_vendor_parity(
            3,
            RawApiId::INT2E_G1_SPH,
            RawApiId::INT2E_G1_CART,
            vendor_ffi::vendor_int2e_g1_sph,
            vendor_ffi::vendor_int2e_g1_cart,
            "int2e_g1",
        );
    }

    #[test]
    fn test_int2e_ig1_parity() {
        giao_2e_vendor_parity(
            3,
            RawApiId::INT2E_IG1_SPH,
            RawApiId::INT2E_IG1_CART,
            vendor_ffi::vendor_int2e_ig1_sph,
            vendor_ffi::vendor_int2e_ig1_cart,
            "int2e_ig1",
        );
    }

    #[test]
    fn test_int2e_gg1_parity() {
        giao_2e_vendor_parity(
            9,
            RawApiId::INT2E_GG1_SPH,
            RawApiId::INT2E_GG1_CART,
            vendor_ffi::vendor_int2e_gg1_sph,
            vendor_ffi::vendor_int2e_gg1_cart,
            "int2e_gg1",
        );
    }

    #[test]
    fn test_int2e_g1g2_parity() {
        // D-16: int2e_g1g2 (gauge factor on BOTH electrons) is IN scope.
        giao_2e_vendor_parity(
            9,
            RawApiId::INT2E_G1G2_SPH,
            RawApiId::INT2E_G1G2_CART,
            vendor_ffi::vendor_int2e_g1g2_sph,
            vendor_ffi::vendor_int2e_g1g2_cart,
            "int2e_g1g2",
        );
    }
}
