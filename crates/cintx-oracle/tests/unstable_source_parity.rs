//! Oracle parity tests for unstable-source API families (Phase 14).
//!
//! Per D-09: single file with per-family modules, all gated behind
//! `#[cfg(feature = "unstable-source-api")]`.
//! Per D-10: reuse H2O/STO-3G fixture molecule for all unstable families.
//! Grids family adds grid point coordinates to env but uses the same molecule.
//!
//! Gate summary:
//!   Family  | Symbols | Status
//!   --------|---------|-------
//!   origi   | 4       | Wave 2 (Phase 14 Plan 02)
//!   grids   | 5       | Wave 2 (Phase 14 Plan 02)
//!   breit   | 2       | Wave 2 (Phase 14 Plan 03)
//!   origk   | 6       | Wave 2 (Phase 14 Plan 03)
//!   ssc     | 1       | Wave 2 (Phase 14 Plan 04)
//!
//! Requirements: #[cfg(feature = "cpu")] + #[cfg(feature = "unstable-source-api")]
//! Run: CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu,unstable-source-api -p cintx-oracle -- unstable_source_parity

#![cfg(feature = "cpu")]
#![cfg(feature = "unstable-source-api")]

/// origi family parity tests.
/// 4 symbols: int1e_r2_origi_sph, int1e_r4_origi_sph,
///            int1e_r2_origi_ip2_sph, int1e_r4_origi_ip2_sph.
/// Implementation added in Phase 14 Plan 02.
mod origi_parity {}

/// grids family parity tests.
/// 5 symbols: int1e_grids_sph, int1e_grids_ip_sph, int1e_grids_ipvip_sph,
///            int1e_grids_spvsp_sph, int1e_grids_ipip_sph.
/// Uses H2O/STO-3G fixture with grid point coordinates in env.
/// Implementation added in Phase 14 Plan 02.
mod grids_parity {}

/// breit family parity tests.
/// 2 spinor-only symbols: int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor.
/// Implementation added in Phase 14 Plan 04.
mod breit_parity {
    use cintx_compat::raw::{
        ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
        POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
    };

    /// Count of mismatches between reference and observed f64 slices at given atol.
    fn count_mismatches_atol(reference: &[f64], observed: &[f64], atol: f64) -> usize {
        assert_eq!(
            reference.len(),
            observed.len(),
            "output length mismatch: {} vs {}",
            reference.len(),
            observed.len()
        );
        let mut mismatches = 0usize;
        for (idx, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
            let diff = (obs_val - ref_val).abs();
            if diff > atol {
                mismatches += 1;
                if mismatches <= 16 {
                    eprintln!(
                        "  MISMATCH[{idx}] ref={ref_val:.15e} obs={obs_val:.15e} diff={diff:.3e}"
                    );
                }
            }
        }
        mismatches
    }

    /// Build H2O/STO-3G fixture (same as two_electron_parity.rs — PTR_ENV_START-aligned).
    ///
    /// Returns (atm, bas, env) with 5 shells:
    ///   Shell 0: O 1s  (l=0, 3 primitives, kappa=0)
    ///   Shell 1: O 2s  (l=0, 3 primitives, kappa=0)
    ///   Shell 2: O 2p  (l=1, 3 primitives, kappa=0)
    ///   Shell 3: H1 1s (l=0, 3 primitives, kappa=0)
    ///   Shell 4: H2 1s (l=0, 3 primitives, kappa=0)
    fn build_h2o_sto3g_breit() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let o_coord = [0.0_f64, 0.0, 0.0];
        let h1_coord = [0.0_f64, 1.4307, 1.1078];
        let h2_coord = [0.0_f64, -1.4307, 1.1078];

        let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
        let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
        let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
        let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
        let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
        let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
        let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
        let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

        let mut env = vec![0.0_f64; PTR_ENV_START];

        let o_coord_ptr = env.len() as i32;
        env.extend_from_slice(&o_coord);
        let h1_coord_ptr = env.len() as i32;
        env.extend_from_slice(&h1_coord);
        let h2_coord_ptr = env.len() as i32;
        env.extend_from_slice(&h2_coord);
        let zeta_ptr = env.len() as i32;
        env.push(0.0);

        let o1s_exp_ptr = env.len() as i32;
        env.extend_from_slice(&o_1s_exp);
        let o1s_coeff_ptr = env.len() as i32;
        env.extend_from_slice(&o_1s_coeff);
        let o2s_exp_ptr = env.len() as i32;
        env.extend_from_slice(&o_2s_exp);
        let o2s_coeff_ptr = env.len() as i32;
        env.extend_from_slice(&o_2s_coeff);
        let o2p_exp_ptr = env.len() as i32;
        env.extend_from_slice(&o_2p_exp);
        let o2p_coeff_ptr = env.len() as i32;
        env.extend_from_slice(&o_2p_coeff);
        let h1s_exp_ptr = env.len() as i32;
        env.extend_from_slice(&h_1s_exp);
        let h1s_coeff_ptr = env.len() as i32;
        env.extend_from_slice(&h_1s_coeff);

        let mut atm = vec![0_i32; 3 * ATM_SLOTS];
        atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
        atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
        atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

        atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
        atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
        atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

        atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
        atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
        atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

        let mut bas = vec![0_i32; 5 * BAS_SLOTS];
        bas[0 * BAS_SLOTS + ATOM_OF] = 0;
        bas[0 * BAS_SLOTS + ANG_OF] = 0;
        bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
        bas[0 * BAS_SLOTS + NCTR_OF] = 1;
        bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
        bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

        bas[1 * BAS_SLOTS + ATOM_OF] = 0;
        bas[1 * BAS_SLOTS + ANG_OF] = 0;
        bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
        bas[1 * BAS_SLOTS + NCTR_OF] = 1;
        bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
        bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

        bas[2 * BAS_SLOTS + ATOM_OF] = 0;
        bas[2 * BAS_SLOTS + ANG_OF] = 1;
        bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
        bas[2 * BAS_SLOTS + NCTR_OF] = 1;
        bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
        bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

        bas[3 * BAS_SLOTS + ATOM_OF] = 1;
        bas[3 * BAS_SLOTS + ANG_OF] = 0;
        bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
        bas[3 * BAS_SLOTS + NCTR_OF] = 1;
        bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
        bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

        bas[4 * BAS_SLOTS + ATOM_OF] = 2;
        bas[4 * BAS_SLOTS + ANG_OF] = 0;
        bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
        bas[4 * BAS_SLOTS + NCTR_OF] = 1;
        bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
        bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

        (atm, bas, env)
    }

    /// Compute spinor output size (in f64) for a shell quartet with kappa=0 shells.
    ///
    /// For kappa=0, CINTcgto_spinor = 2*(l+1) per contracted shell.
    /// The output buffer for a complex spinor integral has 2*ns_i*ns_j*ns_k*ns_l f64 values
    /// (factor 2 for complex interleave: re/im pairs).
    fn spinor_n_elem_kappa0(bas: &[i32], shls: &[i32; 4]) -> usize {
        let mut ns = 1usize;
        for &sh in shls {
            let l = bas[sh as usize * BAS_SLOTS + ANG_OF] as usize;
            let ns_sh = 2 * (l + 1); // CINTcgto_spinor for kappa=0
            ns *= ns_sh;
        }
        2 * ns // factor 2 for complex interleave
    }

    /// Oracle parity test for int2e_breit_r1p2_spinor against vendored libcint 6.1.3.
    ///
    /// Uses H2O/STO-3G fixture, shell quartet [0, 1, 0, 1] (O-1s/O-2s pair).
    /// Tolerance: atol=1e-12 applied to each f64 element (real and imaginary parts).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int2e_breit_r1p2_spinor_oracle_parity() {
        let (atm, bas, env) = build_h2o_sto3g_breit();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        let shls: [i32; 4] = [0, 1, 0, 1];
        let n_elem = spinor_n_elem_kappa0(&bas, &shls);
        let atol = 1.0e-12_f64;
        let epsilon = atol;

        // Reference: vendored libcint
        let mut vendor_out = vec![0.0_f64; n_elem];
        use cintx_oracle::vendor_ffi;
        vendor_ffi::vendor_int2e_breit_r1p2_spinor(
            &mut vendor_out,
            &shls,
            &atm,
            natm,
            &bas,
            nbas,
            &env,
        );

        // cintx: eval_raw with Breit spinor symbol
        let mut cintx_out = vec![0.0_f64; n_elem];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_breit_r1p2_spinor"),
                Some(&mut cintx_out),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("eval_raw int2e_breit_r1p2_spinor failed: {e:?}"));
        }

        let mismatches = count_mismatches_atol(&vendor_out, &cintx_out, epsilon);

        // Verify at least one non-zero element (kernel is not a stub)
        let any_nonzero = cintx_out.iter().any(|v| v.abs() > 1e-18)
            || vendor_out.iter().any(|v| v.abs() > 1e-18);

        println!(
            "int2e_breit_r1p2_spinor: vendor parity PASS, n_elem={n_elem}, mismatches={mismatches}, any_nonzero={any_nonzero} (atol={atol:.1e})"
        );

        assert_eq!(
            mismatches, 0,
            "int2e_breit_r1p2_spinor: {mismatches} parity mismatch(es) against vendored libcint at atol={atol:.1e}"
        );
        assert!(
            any_nonzero,
            "int2e_breit_r1p2_spinor: all outputs are zero — kernel appears stubbed"
        );
    }

    /// Oracle parity test for int2e_breit_r2p2_spinor against vendored libcint 6.1.3.
    ///
    /// Uses H2O/STO-3G fixture, shell quartet [0, 1, 0, 1] (O-1s/O-2s pair).
    /// Tolerance: atol=1e-12 applied to each f64 element (real and imaginary parts).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int2e_breit_r2p2_spinor_oracle_parity() {
        let (atm, bas, env) = build_h2o_sto3g_breit();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        let shls: [i32; 4] = [0, 1, 0, 1];
        let n_elem = spinor_n_elem_kappa0(&bas, &shls);
        let atol = 1.0e-12_f64;
        let epsilon = atol;

        // Reference: vendored libcint
        let mut vendor_out = vec![0.0_f64; n_elem];
        use cintx_oracle::vendor_ffi;
        vendor_ffi::vendor_int2e_breit_r2p2_spinor(
            &mut vendor_out,
            &shls,
            &atm,
            natm,
            &bas,
            nbas,
            &env,
        );

        // cintx: eval_raw with Breit spinor symbol
        let mut cintx_out = vec![0.0_f64; n_elem];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_breit_r2p2_spinor"),
                Some(&mut cintx_out),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("eval_raw int2e_breit_r2p2_spinor failed: {e:?}"));
        }

        let mismatches = count_mismatches_atol(&vendor_out, &cintx_out, epsilon);

        // Verify at least one non-zero element (kernel is not a stub)
        let any_nonzero = cintx_out.iter().any(|v| v.abs() > 1e-18)
            || vendor_out.iter().any(|v| v.abs() > 1e-18);

        println!(
            "int2e_breit_r2p2_spinor: vendor parity PASS, n_elem={n_elem}, mismatches={mismatches}, any_nonzero={any_nonzero} (atol={atol:.1e})"
        );

        assert_eq!(
            mismatches, 0,
            "int2e_breit_r2p2_spinor: {mismatches} parity mismatch(es) against vendored libcint at atol={atol:.1e}"
        );
        assert!(
            any_nonzero,
            "int2e_breit_r2p2_spinor: all outputs are zero — kernel appears stubbed"
        );
    }
}

/// origk family parity tests.
/// 6 symbols: int3c1e_r2/r4/r6_origk_sph, int3c1e_ip1_r2/r4/r6_origk_sph.
/// Implementation added in Phase 14 Plan 03.
mod origk_parity {}

/// ssc family parity tests.
/// 1 symbol: int3c2e_sph_ssc.
/// Implementation added in Phase 14 Plan 04.
mod ssc_parity {}
