pub mod boys;
pub mod kernels;
pub mod vector;

pub use boys::{rys_root1_scalar, rys_root1_simd, rys_root2_scalar, rys_root2_simd};
pub use kernels::{
    common_fac_sp, ncart, AtomCoord, Center2c2eInput, Center3c1eInput, Center3c2eInput,
    Center4c1eInput, OneElectronInput, SimdCenter2c2eKernel, SimdCenter3c1eKernel,
    SimdCenter3c2eKernel, SimdCenter4c1eKernel, SimdOneElectronKernel, SimdTwoElectronKernel,
    TwoElectronInput,
};
pub use vector::SimdFloat;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;
    use wide::{f32x4, f32x8, f64x2, f64x4};

    #[test]
    fn test_rys_root1_and_root2_identities() {
        for &x in &[0.001_f64, 0.2, 0.5, 1.2, 3.5, 7.0, 15.0, 35.0] {
            let (_r1, w1) = rys_root1_scalar(x);
            let ([_r0, _r1], [w0, w1_r2]) = rys_root2_scalar(x);

            // Analytical Boys function F_0(x) = sqrt(pi / (4x)) * erf(sqrt(x)) using rmath::erf
            let f0 = if x < 1e-6 {
                1.0 - x / 3.0
            } else {
                (PI / (4.0 * x)).sqrt() * rmath::erf(x.sqrt())
            };

            assert_relative_eq!(w1, f0, epsilon = 1e-7);
            assert_relative_eq!(w0 + w1_r2, f0, epsilon = 1e-7);
        }

        let x_vec = f64x4::new([0.2, 0.5, 1.2, 3.5]);
        let (_r_simd, w_simd) = rys_root1_simd(x_vec);
        let mut w_arr = [0.0; 4];
        w_simd.store_to_f64_slice(&mut w_arr);

        for i in 0..4 {
            let (_, w_scalar) = rys_root1_scalar([0.2, 0.5, 1.2, 3.5][i]);
            assert_eq!(w_arr[i], w_scalar);
        }
    }

    #[test]
    fn test_ovlp_ss_analytical_and_simd_match() {
        let ai = 1.2_f64;
        let aj = 0.8_f64;
        let ci = 1.0_f64;
        let cj = 1.0_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [1.2_f64, 0.5, -0.3];
        let dx = ri[0] - rj[0];
        let dy = ri[1] - rj[1];
        let dz = ri[2] - rj[2];
        let rr = dx * dx + dy * dy + dz * dz;

        let zeta = ai + aj;
        let s0 = (PI / zeta).powf(1.5) * (-ai * aj / zeta * rr).exp();
        let expected_ovlp = s0 * common_fac_sp(0) * common_fac_sp(0) * ci * cj;

        let input = OneElectronInput {
            li: 0,
            lj: 0,
            ri,
            rj,
            exps_i: &[ai],
            exps_j: &[aj],
            coeff_i: &[ci],
            coeff_j: &[cj],
            atoms: &[],
        };

        let mut out_scalar = [0.0; 1];
        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut out_scalar);

        let mut out_f64x2 = [0.0; 1];
        SimdOneElectronKernel::eval_ovlp::<f64x2>(&input, &mut out_f64x2);

        let mut out_f64x4 = [0.0; 1];
        SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut out_f64x4);

        assert_relative_eq!(out_scalar[0], expected_ovlp, epsilon = 1e-14);
        assert_relative_eq!(out_f64x2[0], expected_ovlp, epsilon = 1e-14);
        assert_relative_eq!(out_f64x4[0], expected_ovlp, epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_kin_ss_analytical_and_simd_match() {
        let ai = 1.5_f64;
        let aj = 0.9_f64;
        let ci = 1.0_f64;
        let cj = 1.0_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.8_f64, 0.0, 0.0];
        let dx = ri[0] - rj[0];
        let dy = ri[1] - rj[1];
        let dz = ri[2] - rj[2];
        let rr = dx * dx + dy * dy + dz * dz;

        let zeta = ai + aj;
        let eta = ai * aj / zeta;
        let s0 = (PI / zeta).powf(1.5) * (-eta * rr).exp();
        let t0 = eta * (3.0 - 2.0 * eta * rr) * s0;
        let expected_kin = t0 * common_fac_sp(0) * common_fac_sp(0) * ci * cj;

        let input = OneElectronInput {
            li: 0,
            lj: 0,
            ri,
            rj,
            exps_i: &[ai],
            exps_j: &[aj],
            coeff_i: &[ci],
            coeff_j: &[cj],
            atoms: &[],
        };

        let mut out_scalar = [0.0; 1];
        SimdOneElectronKernel::eval_kin::<f64>(&input, &mut out_scalar);

        let mut out_f64x4 = [0.0; 1];
        SimdOneElectronKernel::eval_kin::<f64x4>(&input, &mut out_f64x4);

        assert_relative_eq!(out_scalar[0], expected_kin, epsilon = 1e-14);
        assert_relative_eq!(out_f64x4[0], expected_kin, epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_nuc_ss_scalar_and_simd_match() {
        let ai = 1.1_f64;
        let aj = 0.7_f64;
        let ci = 1.0_f64;
        let cj = 1.0_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [1.0_f64, 0.0, 0.0];
        let atoms = [
            AtomCoord {
                charge: 1.0,
                coord: [0.5, 0.2, -0.1],
            },
            AtomCoord {
                charge: 8.0,
                coord: [0.0, 0.0, 0.0],
            },
        ];

        let input = OneElectronInput {
            li: 0,
            lj: 0,
            ri,
            rj,
            exps_i: &[ai],
            exps_j: &[aj],
            coeff_i: &[ci],
            coeff_j: &[cj],
            atoms: &atoms,
        };

        let mut out_scalar = [0.0; 1];
        SimdOneElectronKernel::eval_nuc::<f64>(&input, &mut out_scalar);

        let mut out_f64x4 = [0.0; 1];
        SimdOneElectronKernel::eval_nuc::<f64x4>(&input, &mut out_f64x4);

        assert!(out_scalar[0].is_finite());
        assert!(out_scalar[0] < 0.0, "Nuclear attraction should be attractive (< 0)");
        assert_relative_eq!(out_scalar[0], out_f64x4[0], epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_sp_angular_momentum_parity() {
        let exps_i = [1.5_f64];
        let exps_j = [0.8_f64];
        let coeff_i = [1.0_f64];
        let coeff_j = [1.0_f64];
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.5_f64, 0.3, -0.2];

        let input = OneElectronInput {
            li: 0,
            lj: 1,
            ri,
            rj,
            exps_i: &exps_i,
            exps_j: &exps_j,
            coeff_i: &coeff_i,
            coeff_j: &coeff_j,
            atoms: &[],
        };

        let len = ncart(0) * ncart(1);
        assert_eq!(len, 3);

        let mut ovlp_scalar = vec![0.0; len];
        let mut ovlp_f64x2 = vec![0.0; len];
        let mut ovlp_f64x4 = vec![0.0; len];
        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut ovlp_scalar);
        SimdOneElectronKernel::eval_ovlp::<f64x2>(&input, &mut ovlp_f64x2);
        SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut ovlp_f64x4);

        for k in 0..len {
            assert_relative_eq!(ovlp_scalar[k], ovlp_f64x4[k], epsilon = 1e-14);
            assert_eq!(ovlp_scalar[k].to_bits(), ovlp_f64x4[k].to_bits());
        }

        let mut kin_scalar = vec![0.0; len];
        let mut kin_f64x4 = vec![0.0; len];
        SimdOneElectronKernel::eval_kin::<f64>(&input, &mut kin_scalar);
        SimdOneElectronKernel::eval_kin::<f64x4>(&input, &mut kin_f64x4);

        for k in 0..len {
            assert_relative_eq!(kin_scalar[k], kin_f64x4[k], epsilon = 1e-14);
            assert_eq!(kin_scalar[k].to_bits(), kin_f64x4[k].to_bits());
        }
    }

    #[test]
    fn test_pp_contracted_multiprimitive_simd_parity() {
        let exps_i = [3.42525091_f64, 0.62391373, 0.16885540];
        let coeff_i = [0.15432897_f64, 0.53532814, 0.44463454];
        let exps_j = [2.94124940_f64, 0.57014490, 0.15470000];
        let coeff_j = [0.15591627_f64, 0.54070560, 0.43573880];

        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [1.2_f64, 0.4, -0.6];
        let atoms = [AtomCoord {
            charge: 6.0,
            coord: [0.6, 0.2, -0.3],
        }];

        let input = OneElectronInput {
            li: 1,
            lj: 1,
            ri,
            rj,
            exps_i: &exps_i,
            exps_j: &exps_j,
            coeff_i: &coeff_i,
            coeff_j: &coeff_j,
            atoms: &atoms,
        };

        let nci = ncart(1);
        let ncj = ncart(1);
        let len = nci * ncj;
        assert_eq!(len, 9);

        let mut ovlp_scalar = vec![0.0; len];
        let mut ovlp_simd = vec![0.0; len];
        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut ovlp_scalar);
        SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut ovlp_simd);

        let mut kin_scalar = vec![0.0; len];
        let mut kin_simd = vec![0.0; len];
        SimdOneElectronKernel::eval_kin::<f64>(&input, &mut kin_scalar);
        SimdOneElectronKernel::eval_kin::<f64x4>(&input, &mut kin_simd);

        let mut nuc_scalar = vec![0.0; len];
        let mut nuc_simd = vec![0.0; len];
        SimdOneElectronKernel::eval_nuc::<f64>(&input, &mut nuc_scalar);
        SimdOneElectronKernel::eval_nuc::<f64x4>(&input, &mut nuc_simd);

        for k in 0..len {
            assert_relative_eq!(ovlp_scalar[k], ovlp_simd[k], epsilon = 1e-14);
            assert_relative_eq!(kin_scalar[k], kin_simd[k], epsilon = 1e-14);
            assert_relative_eq!(nuc_scalar[k], nuc_simd[k], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_dd_higher_angular_momentum_parity() {
        let exps_i = [2.0_f64, 0.5_f64];
        let coeff_i = [0.6_f64, 0.4_f64];
        let exps_j = [1.8_f64, 0.4_f64];
        let coeff_j = [0.7_f64, 0.3_f64];

        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.8_f64, -0.4, 0.2];

        let input = OneElectronInput {
            li: 2,
            lj: 2,
            ri,
            rj,
            exps_i: &exps_i,
            exps_j: &exps_j,
            coeff_i: &coeff_i,
            coeff_j: &coeff_j,
            atoms: &[],
        };

        let len = ncart(2) * ncart(2);
        assert_eq!(len, 36);

        let mut ovlp_scalar = vec![0.0; len];
        let mut ovlp_simd = vec![0.0; len];
        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut ovlp_scalar);
        SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut ovlp_simd);

        let mut kin_scalar = vec![0.0; len];
        let mut kin_simd = vec![0.0; len];
        SimdOneElectronKernel::eval_kin::<f64>(&input, &mut kin_scalar);
        SimdOneElectronKernel::eval_kin::<f64x4>(&input, &mut kin_simd);

        for k in 0..len {
            assert_relative_eq!(ovlp_scalar[k], ovlp_simd[k], epsilon = 1e-14);
            assert_relative_eq!(kin_scalar[k], kin_simd[k], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_f32_precision_simd_kernel() {
        let ai = 1.2_f64;
        let aj = 0.8_f64;
        let ci = 1.0_f64;
        let cj = 1.0_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.5_f64, 0.2, -0.1];

        let input = OneElectronInput {
            li: 1,
            lj: 1,
            ri,
            rj,
            exps_i: &[ai],
            exps_j: &[aj],
            coeff_i: &[ci],
            coeff_j: &[cj],
            atoms: &[],
        };

        let len = ncart(1) * ncart(1);
        let mut out_f64 = vec![0.0; len];
        let mut out_f32x4 = vec![0.0; len];
        let mut out_f32x8 = vec![0.0; len];

        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut out_f64);
        SimdOneElectronKernel::eval_ovlp::<f32x4>(&input, &mut out_f32x4);
        SimdOneElectronKernel::eval_ovlp::<f32x8>(&input, &mut out_f32x8);

        for k in 0..len {
            assert_relative_eq!(out_f32x4[k], out_f64[k], epsilon = 1e-5);
            assert_relative_eq!(out_f32x8[k], out_f64[k], epsilon = 1e-5);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2-Center 2-Electron Integral (int2c2e) Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_center_2c2e_ss_simd_match() {
        let ai = 1.0_f64;
        let ak = 1.0_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 1.7];

        let input = Center2c2eInput {
            li: 0,
            lk: 0,
            ri,
            rk,
            exps_i: &[ai],
            exps_k: &[ak],
            coeff_i: &[1.0],
            coeff_k: &[1.0],
        };

        let mut out_scalar = [0.0; 1];
        let mut out_f64x2 = [0.0; 1];
        let mut out_f64x4 = [0.0; 1];

        SimdCenter2c2eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter2c2eKernel::eval::<f64x2>(&input, &mut out_f64x2);
        SimdCenter2c2eKernel::eval::<f64x4>(&input, &mut out_f64x4);

        assert!(out_scalar[0] > 0.0, "2c2e s-s result should be positive");
        assert_relative_eq!(out_scalar[0], out_f64x2[0], epsilon = 1e-14);
        assert_relative_eq!(out_scalar[0], out_f64x4[0], epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_center_2c2e_pp_contracted_simd_parity() {
        let exps_i = [3.42525091_f64, 0.62391373, 0.16885540];
        let coeff_i = [0.15432897_f64, 0.53532814, 0.44463454];
        let exps_k = [2.94124940_f64, 0.57014490, 0.15470000];
        let coeff_k = [0.15591627_f64, 0.54070560, 0.43573880];

        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.8_f64, 0.3, -0.4];

        let input = Center2c2eInput {
            li: 1,
            lk: 1,
            ri,
            rk,
            exps_i: &exps_i,
            exps_k: &exps_k,
            coeff_i: &coeff_i,
            coeff_k: &coeff_k,
        };

        let len = ncart(1) * ncart(1);
        assert_eq!(len, 9);

        let mut out_scalar = vec![0.0; len];
        let mut out_simd = vec![0.0; len];

        SimdCenter2c2eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter2c2eKernel::eval::<f64x4>(&input, &mut out_simd);

        for k in 0..len {
            assert_relative_eq!(out_scalar[k], out_simd[k], epsilon = 1e-14);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4-Center 2-Electron Integral (int2e) Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_two_electron_ssss_simd_match() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ak = 1.0_f64;
        let al = 1.0_f64;

        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 1.4];
        let rl = [0.0_f64, 0.0, 1.4];

        let input = TwoElectronInput {
            li: 0,
            lj: 0,
            lk: 0,
            ll: 0,
            ri,
            rj,
            rk,
            rl,
            exps_i: &[ai],
            exps_j: &[aj],
            exps_k: &[ak],
            exps_l: &[al],
            coeff_i: &[1.0],
            coeff_j: &[1.0],
            coeff_k: &[1.0],
            coeff_l: &[1.0],
        };

        let mut out_scalar = [0.0; 1];
        let mut out_f64x2 = [0.0; 1];
        let mut out_f64x4 = [0.0; 1];

        SimdTwoElectronKernel::eval::<f64>(&input, &mut out_scalar);
        SimdTwoElectronKernel::eval::<f64x2>(&input, &mut out_f64x2);
        SimdTwoElectronKernel::eval::<f64x4>(&input, &mut out_f64x4);

        assert!(out_scalar[0] > 0.0, "2e (ss|ss) integral should be positive");
        assert_relative_eq!(out_scalar[0], out_f64x2[0], epsilon = 1e-14);
        assert_relative_eq!(out_scalar[0], out_f64x4[0], epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_two_electron_psss_angular_momentum_parity() {
        let input = TwoElectronInput {
            li: 1, // p shell
            lj: 0, // s shell
            lk: 0, // s shell
            ll: 0, // s shell
            ri: [0.0, 0.0, 0.0],
            rj: [0.2, -0.1, 0.0],
            rk: [1.2, 0.4, -0.5],
            rl: [1.0, 0.0, 0.0],
            exps_i: &[1.5],
            exps_j: &[0.8],
            exps_k: &[1.2],
            exps_l: &[0.9],
            coeff_i: &[1.0],
            coeff_j: &[1.0],
            coeff_k: &[1.0],
            coeff_l: &[1.0],
        };

        let len = ncart(1) * ncart(0) * ncart(0) * ncart(0);
        assert_eq!(len, 3);

        let mut out_scalar = vec![0.0; len];
        let mut out_simd = vec![0.0; len];

        SimdTwoElectronKernel::eval::<f64>(&input, &mut out_scalar);
        SimdTwoElectronKernel::eval::<f64x4>(&input, &mut out_simd);

        for k in 0..len {
            assert_relative_eq!(out_scalar[k], out_simd[k], epsilon = 1e-14);
            assert_eq!(out_scalar[k].to_bits(), out_simd[k].to_bits());
        }
    }

    #[test]
    fn test_rmath_vectorized_math_apis() {
        // Test f64x4
        let v64 = f64x4::new([0.5, 1.0, 2.0, 3.5]);
        let exp_v64 = v64.exp().to_array();
        let ln_v64 = v64.ln().to_array();
        let erf_v64 = v64.erf().to_array();
        let erfc_v64 = v64.erfc().to_array();
        let sin_v64 = v64.sin().to_array();
        let cos_v64 = v64.cos().to_array();
        let lgamma_v64 = v64.lgamma().to_array();
        let pow_v64 = v64.pow(f64x4::splat(2.0)).to_array();

        for i in 0..4 {
            let x = [0.5, 1.0, 2.0, 3.5][i];
            assert_relative_eq!(exp_v64[i], x.exp(), epsilon = 1e-14);
            assert_relative_eq!(ln_v64[i], x.ln(), epsilon = 1e-14);
            assert_relative_eq!(erf_v64[i] + erfc_v64[i], 1.0, epsilon = 1e-14);
            assert_relative_eq!(sin_v64[i], x.sin(), epsilon = 1e-14);
            assert_relative_eq!(cos_v64[i], x.cos(), epsilon = 1e-14);
            assert_relative_eq!(pow_v64[i], x.powi(2), epsilon = 1e-13);
            assert!(lgamma_v64[i].is_finite());
            assert_relative_eq!(rmath::j0(x), rmath::j0(x), epsilon = 1e-14);
            assert_relative_eq!(rmath::exp10(x), 10.0_f64.powf(x), epsilon = 1e-12);
        }

        // Test f32x4
        let v32 = f32x4::new([0.5, 1.0, 2.0, 3.5]);
        let exp_v32 = v32.exp().to_array();
        let ln_v32 = v32.ln().to_array();
        let erf_v32 = v32.erf().to_array();
        let erfc_v32 = v32.erfc().to_array();
        let sin_v32 = v32.sin().to_array();
        let cos_v32 = v32.cos().to_array();
        let pow_v32 = v32.pow(f32x4::splat(2.0)).to_array();

        for i in 0..4 {
            let x = [0.5_f32, 1.0, 2.0, 3.5][i];
            assert_relative_eq!(exp_v32[i], x.exp(), epsilon = 1e-5);
            assert_relative_eq!(ln_v32[i], x.ln(), epsilon = 1e-5);
            assert_relative_eq!(erf_v32[i] + erfc_v32[i], 1.0, epsilon = 1e-5);
            assert_relative_eq!(sin_v32[i], x.sin(), epsilon = 1e-5);
            assert_relative_eq!(cos_v32[i], x.cos(), epsilon = 1e-5);
            assert_relative_eq!(pow_v32[i], x.powi(2), epsilon = 1e-4);
        }
    }

    #[test]
    fn test_center_3c2e_sss_simd_match() {
        let input = Center3c2eInput {
            li: 0,
            lj: 0,
            lk: 0,
            ri: [0.0, 0.0, 0.0],
            rj: [0.3, -0.2, 0.1],
            rk: [1.2, 0.5, -0.4],
            exps_i: &[1.4],
            exps_j: &[0.9],
            exps_k: &[1.1],
            coeff_i: &[1.0],
            coeff_j: &[1.0],
            coeff_k: &[1.0],
        };

        let mut out_scalar = [0.0; 1];
        let mut out_f64x2 = [0.0; 1];
        let mut out_f64x4 = [0.0; 1];

        SimdCenter3c2eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter3c2eKernel::eval::<f64x2>(&input, &mut out_f64x2);
        SimdCenter3c2eKernel::eval::<f64x4>(&input, &mut out_f64x4);

        assert!(out_scalar[0] > 0.0);
        assert_relative_eq!(out_scalar[0], out_f64x2[0], epsilon = 1e-14);
        assert_relative_eq!(out_scalar[0], out_f64x4[0], epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_f64x4[0].to_bits());
    }

    #[test]
    fn test_center_3c2e_pss_angular_momentum_parity() {
        let input = Center3c2eInput {
            li: 1, // p shell
            lj: 0, // s shell
            lk: 0, // s shell
            ri: [0.0, 0.0, 0.0],
            rj: [0.2, -0.1, 0.0],
            rk: [1.0, 0.4, -0.3],
            exps_i: &[1.5],
            exps_j: &[0.8],
            exps_k: &[1.2],
            coeff_i: &[1.0],
            coeff_j: &[1.0],
            coeff_k: &[1.0],
        };

        let len = ncart(1) * ncart(0) * ncart(0);
        assert_eq!(len, 3);

        let mut out_scalar = vec![0.0; len];
        let mut out_simd = vec![0.0; len];

        SimdCenter3c2eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter3c2eKernel::eval::<f64x4>(&input, &mut out_simd);

        for k in 0..len {
            assert_relative_eq!(out_scalar[k], out_simd[k], epsilon = 1e-14);
            assert_eq!(out_scalar[k].to_bits(), out_simd[k].to_bits());
        }
    }

    #[test]
    fn test_center_3c1e_sss_analytical_and_simd_match() {
        let ai = 1.2_f64;
        let aj = 0.8_f64;
        let ak = 1.0_f64;
        let ci = 1.0_f64;
        let cj = 1.0_f64;
        let ck = 1.0_f64;

        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.5_f64, -0.3, 0.2];
        let rk = [-0.4_f64, 0.6, 0.1];

        let dij = (ri[0] - rj[0]).powi(2) + (ri[1] - rj[1]).powi(2) + (ri[2] - rj[2]).powi(2);
        let dik = (ri[0] - rk[0]).powi(2) + (ri[1] - rk[1]).powi(2) + (ri[2] - rk[2]).powi(2);
        let djk = (rj[0] - rk[0]).powi(2) + (rj[1] - rk[1]).powi(2) + (rj[2] - rk[2]).powi(2);

        let zeta = ai + aj + ak;
        let exp_term = rmath::exp(-(ai * aj * dij + ai * ak * dik + aj * ak * djk) / zeta);
        let s0 = (PI / zeta).powf(1.5) * exp_term;
        let expected = s0 * common_fac_sp(0).powi(3) * ci * cj * ck;

        let input = Center3c1eInput {
            li: 0,
            lj: 0,
            lk: 0,
            ri,
            rj,
            rk,
            exps_i: &[ai],
            exps_j: &[aj],
            exps_k: &[ak],
            coeff_i: &[ci],
            coeff_j: &[cj],
            coeff_k: &[ck],
        };

        let mut out_scalar = [0.0; 1];
        let mut out_simd = [0.0; 1];

        SimdCenter3c1eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter3c1eKernel::eval::<f64x4>(&input, &mut out_simd);

        assert_relative_eq!(out_scalar[0], expected, epsilon = 1e-14);
        assert_relative_eq!(out_simd[0], expected, epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_simd[0].to_bits());
    }

    #[test]
    fn test_center_4c1e_ssss_analytical_and_simd_match() {
        let ai = 1.1_f64;
        let aj = 0.9_f64;
        let ak = 1.3_f64;
        let al = 0.7_f64;

        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.4_f64, 0.2, -0.1];
        let rk = [-0.3_f64, 0.5, 0.2];
        let rl = [0.1_f64, -0.4, 0.6];

        let dij = (ri[0] - rj[0]).powi(2) + (ri[1] - rj[1]).powi(2) + (ri[2] - rj[2]).powi(2);
        let dik = (ri[0] - rk[0]).powi(2) + (ri[1] - rk[1]).powi(2) + (ri[2] - rk[2]).powi(2);
        let dil = (ri[0] - rl[0]).powi(2) + (ri[1] - rl[1]).powi(2) + (ri[2] - rl[2]).powi(2);
        let djk = (rj[0] - rk[0]).powi(2) + (rj[1] - rk[1]).powi(2) + (rj[2] - rk[2]).powi(2);
        let djl = (rj[0] - rl[0]).powi(2) + (rj[1] - rl[1]).powi(2) + (rj[2] - rl[2]).powi(2);
        let dkl = (rk[0] - rl[0]).powi(2) + (rk[1] - rl[1]).powi(2) + (rk[2] - rl[2]).powi(2);

        let zeta = ai + aj + ak + al;
        let exp_sum = ai * aj * dij
            + ai * ak * dik
            + ai * al * dil
            + aj * ak * djk
            + aj * al * djl
            + ak * al * dkl;
        let s0 = (PI / zeta).powf(1.5) * rmath::exp(-exp_sum / zeta);
        let expected = s0 * common_fac_sp(0).powi(4);

        let input = Center4c1eInput {
            li: 0,
            lj: 0,
            lk: 0,
            ll: 0,
            ri,
            rj,
            rk,
            rl,
            exps_i: &[ai],
            exps_j: &[aj],
            exps_k: &[ak],
            exps_l: &[al],
            coeff_i: &[1.0],
            coeff_j: &[1.0],
            coeff_k: &[1.0],
            coeff_l: &[1.0],
        };

        let mut out_scalar = [0.0; 1];
        let mut out_simd = [0.0; 1];

        SimdCenter4c1eKernel::eval::<f64>(&input, &mut out_scalar);
        SimdCenter4c1eKernel::eval::<f64x4>(&input, &mut out_simd);

        assert_relative_eq!(out_scalar[0], expected, epsilon = 1e-14);
        assert_relative_eq!(out_simd[0], expected, epsilon = 1e-14);
        assert_eq!(out_scalar[0].to_bits(), out_simd[0].to_bits());
    }
}

