//! Rys quadrature validation tests for `rys_roots` nroots=1..5.
//!
//! Tests use the CubeCL CPU backend to exercise the `#[cube]` polynomial fit functions.
//! Reference values are computed by a host-side evaluator that directly evaluates the
//! same polynomial coefficients from rys_roots.c — serving as a "second opinion" on
//! the Horner evaluation in the cube functions.
//!
//! Tolerance strategy: absolute only, 1e-12 atol per D-19.
//!
//! Traceability (D-16): each test case cites the relevant rys_roots.c source segment.

#[cfg(feature = "cpu")]
mod rys_cpu_tests {
    use cubecl::cpu::CpuRuntime;
    use cubecl::prelude::*;
    use cintx_cubecl::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};

    // ─────────────────────────────────────────────────────────────────────────
    //  Host-side reference evaluators (pure Rust, not #[cube])
    //  Mirror polynomial coefficients from rys_roots.c for validation.
    // ─────────────────────────────────────────────────────────────────────────

    const PIE4: f64 = 0.78539816339744827900_f64;

    /// Horner polynomial evaluation helper.
    fn horner(x: f64, coeffs: &[f64]) -> f64 {
        let mut result = 0.0_f64;
        for &c in coeffs.iter() {
            result = result * x + c;
        }
        result
    }

    /// Host-side rys_root1 reference.
    /// Source: rys_roots.c lines 267-328.
    fn ref_rys_root1(x: f64) -> (f64, f64) {
        if x > 33.0 {
            return (0.5 / (x - 0.5), f64::sqrt(PIE4 / x));
        } else if x < 3.0e-7 {
            return (0.5 - x / 5.0, 1.0 - x / 3.0);
        }
        let e = (-x).exp();
        let f1;
        if x > 15.0 {
            let y = 1.0 / x;
            f1 = (horner(y, &[1.9623264149430e-01, -4.9695241464490e-01, -6.0156581186481e-05])
                * e + f64::sqrt(PIE4 / x) - e) * y * 0.5;
        } else if x > 10.0 {
            let y = 1.0 / x;
            f1 = (horner(y, &[-1.8784686463512e-01, 2.2991849164985e-01, -4.9893752514047e-01, -2.1916512131607e-05])
                * e + f64::sqrt(PIE4 / x) - e) * y * 0.5;
        } else if x > 5.0 {
            let y = 1.0 / x;
            f1 = (horner(y, &[4.6897511375022e-01, -6.9955602298985e-01, 5.3689283271887e-01,
                -3.2883030418398e-01, 2.4645596956002e-01, -4.9984072848436e-01, -3.1501078774085e-06])
                * e + f64::sqrt(PIE4 / x) - e) * y * 0.5;
        } else if x > 3.0 {
            let y = x - 4.0;
            f1 = horner(y, &[-2.62453564772299e-11, 3.24031041623823e-10, -3.614965656163e-09,
                3.760256799971e-08, -3.553558319675e-07, 3.022556449731e-06, -2.290098979647e-05,
                1.526537461148e-04, -8.81947375894379e-04, 4.33207949514611e-03,
                -1.75257821619926e-02, 5.28406320615584e-02]);
        } else if x > 1.0 {
            let y = x - 2.0;
            f1 = horner(y, &[-1.61702782425558e-10, 1.96215250865776e-09, -2.14234468198419e-08,
                2.17216556336318e-07, -1.98850171329371e-06, 1.62429321438911e-05,
                -1.16740298039895e-04, 7.24888732052332e-04, -3.79490003707156e-03,
                1.61723488664661e-02, -5.29428148329736e-02, 1.15702180856167e-01]);
        } else {
            f1 = horner(x, &[-8.36313918003957e-08, 1.21222603512827e-06, -1.15662609053481e-05,
                9.25197374512647e-05, -6.40994113129432e-04, 3.78787044215009e-03,
                -1.85185172458485e-02, 7.14285713298222e-02, -1.99999999997023e-01,
                3.33333333333318e-01]);
        }
        let ww1 = 2.0 * x * f1 + e;
        (f1 / (ww1 - f1), ww1)
    }

    /// Host-side rys_root2 reference.
    /// Source: rys_roots.c lines 330-487.
    fn ref_rys_root2(x: f64) -> ([f64; 2], [f64; 2]) {
        let r12 = 2.75255128608411e-01_f64;
        let r22 = 2.72474487139158e+00_f64;
        let w22 = 9.17517095361369e-02_f64;

        if x >= 40.0 {
            let ww1_b = f64::sqrt(PIE4 / x);
            let ww2 = w22 * ww1_b;
            return ([r12 / (x - r12), r22 / (x - r22)], [ww1_b - ww2, ww2]);
        }
        if x < 3.0e-7 {
            let rt1 = 1.30693606237085e-01 - 2.90430236082028e-02 * x;
            let rt2 = 2.86930639376291e+00 - 6.37623643058102e-01 * x;
            let ww1 = 6.52145154862545e-01 - 1.22713621927067e-01 * x;
            let ww2 = 3.47854845137453e-01 - 2.10619711404725e-01 * x;
            return ([rt1, rt2], [ww1, ww2]);
        }

        let (y, is_mid) = if x < 3.0 { (x - 2.0, true) } else { (x - 4.0, false) };
        let f1 = if is_mid {
            horner(y, &[-1.61702782425558e-10, 1.96215250865776e-09, -2.14234468198419e-08,
                2.17216556336318e-07, -1.98850171329371e-06, 1.62429321438911e-05,
                -1.16740298039895e-04, 7.24888732052332e-04, -3.79490003707156e-03,
                1.61723488664661e-02, -5.29428148329736e-02, 1.15702180856167e-01])
        } else {
            horner(y, &[-2.62453564772299e-11, 3.24031041623823e-10, -3.614965656163e-09,
                3.760256799971e-08, -3.553558319675e-07, 3.022556449731e-06, -2.290098979647e-05,
                1.526537461148e-04, -8.81947375894379e-04, 4.33207949514611e-03,
                -1.75257821619926e-02, 5.28406320615584e-02])
        };
        let e = (-x).exp();
        let ww1 = (x + x) * f1 + e;
        let (rt1, rt2) = if is_mid {
            let rt1 = horner(y, &[-6.36859636616415e-12, 8.47417064776270e-11, -5.152207846962e-10,
                -3.846389873308e-10, 8.472253388380e-08, -1.85306035634293e-06,
                2.47191693238413e-05, -2.49018321709815e-04, 2.19173220020161e-03,
                -1.63329339286794e-02, 8.68085688285261e-02]);
            let rt2 = horner(y, &[1.45331350488343e-10, 2.07111465297976e-09, -1.878920917404e-08,
                -1.725838516261e-07, 2.247389642339e-06, 9.76783813082564e-06,
                -1.93160765581969e-04, -1.58064140671893e-03, 4.85928174507904e-02,
                -4.30761584997596e-01, 1.80400974537950e+00]);
            (rt1, rt2)
        } else {
            let rt1 = horner(y, &[-4.11560117487296e-12, 7.10910223886747e-11, -1.73508862390291e-09,
                5.93066856324744e-08, -9.76085576741771e-07, 1.08484384385679e-05,
                -1.12608004981982e-04, 1.16210907653515e-03, -9.89572595720351e-03,
                6.12589701086408e-02]);
            let rt2 = horner(y, &[-1.80555625241001e-10, 5.44072475994123e-10, 1.603498045240e-08,
                -1.497986283037e-07, -7.017002532106e-07, 1.85882653064034e-05,
                -2.04685420150802e-05, -2.49327728643089e-03, 3.56550690684281e-02,
                -2.60417417692375e-01, 1.12155283108289e+00]);
            (rt1, rt2)
        };
        let ww2 = ((f1 - ww1) * rt1 + f1) * (1.0 + rt2) / (rt2 - rt1);
        ([rt1, rt2], [ww1 - ww2, ww2])
    }

    /// Host-side rys_root3 asymptotic reference (x >= 3).
    /// Source: rys_roots.c rys_root3() last else branch.
    fn ref_rys_root3_asym(x: f64) -> ([f64; 3], [f64; 3]) {
        let r13 = 1.90163509193487e-01_f64;
        let r23 = 1.78449274854325e+00_f64;
        let w23 = 1.77231492083829e-01_f64;
        let r33 = 5.52534374226326e+00_f64;
        let w33 = 5.11156880411248e-03_f64;
        let ww1_b = f64::sqrt(PIE4 / x);
        let ww3 = w33 * ww1_b;
        let ww2 = w23 * ww1_b;
        let ww1 = ww1_b - ww2 - ww3;
        ([r13 / (x - r13), r23 / (x - r23), r33 / (x - r33)], [ww1, ww2, ww3])
    }

    /// Host-side rys_root5 asymptotic reference (x >= 1).
    /// Source: rys_roots.c rys_root5() last else branch (lines ~1604-1616).
    fn ref_rys_root5_asym(x: f64) -> ([f64; 5], [f64; 5]) {
        let r15 = 1.17581320211778e-01_f64;
        let r25 = 1.07456201243690e+00_f64;
        let w25 = 2.70967405960535e-01_f64;
        let r35 = 3.08593744371754e+00_f64;
        let w35 = 3.82231610015404e-02_f64;
        let r45 = 6.41472973366203e+00_f64;
        let w45 = 1.51614186862443e-03_f64;
        let r55 = 1.18071894899717e+01_f64;
        let w55 = 8.62130526143657e-06_f64;
        let ww1_b = f64::sqrt(PIE4 / x);
        let ww5 = w55 * ww1_b;
        let ww4 = w45 * ww1_b;
        let ww3 = w35 * ww1_b;
        let ww2 = w25 * ww1_b;
        let ww1 = ww1_b - ww2 - ww3 - ww4 - ww5;
        (
            [r15 / (x - r15), r25 / (x - r25), r35 / (x - r35), r45 / (x - r45), r55 / (x - r55)],
            [ww1, ww2, ww3, ww4, ww5],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  CubeCL CPU backend kernels — one per nroots, calling rys_rootN directly.
    //  This avoids a runtime nroots dispatch which causes MLIR index-type issues
    //  in the CubeCL 0.9 CPU backend's LLVM lowering pass.
    // ─────────────────────────────────────────────────────────────────────────

    #[cube(launch)]
    fn rys_root1_kernel(u_out: &mut Array<f64>, w_out: &mut Array<f64>, x: f64) {
        if UNIT_POS == 0 {
            rys_root1(x, u_out, w_out);
        }
    }

    #[cube(launch)]
    fn rys_root2_kernel(u_out: &mut Array<f64>, w_out: &mut Array<f64>, x: f64) {
        if UNIT_POS == 0 {
            rys_root2(x, u_out, w_out);
        }
    }

    #[cube(launch)]
    fn rys_root3_kernel(u_out: &mut Array<f64>, w_out: &mut Array<f64>, x: f64) {
        if UNIT_POS == 0 {
            rys_root3(x, u_out, w_out);
        }
    }

    #[cube(launch)]
    fn rys_root4_kernel(u_out: &mut Array<f64>, w_out: &mut Array<f64>, x: f64) {
        if UNIT_POS == 0 {
            rys_root4(x, u_out, w_out);
        }
    }

    #[cube(launch)]
    fn rys_root5_kernel(u_out: &mut Array<f64>, w_out: &mut Array<f64>, x: f64) {
        if UNIT_POS == 0 {
            rys_root5(x, u_out, w_out);
        }
    }

    /// Evaluate `rys_rootN` via the CubeCL CPU backend.
    /// Returns (roots[0..nroots], weights[0..nroots]).
    fn eval_rys_cpu(nroots: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
        let client = CpuRuntime::client(&Default::default());
        let n = nroots as usize;
        let zeros = vec![0.0f64; n];
        let u_handle = client.create_from_slice(f64::as_bytes(&zeros));
        let w_handle = client.create_from_slice(f64::as_bytes(&zeros));

        let cube_count = CubeCount::Static(1, 1, 1);
        let cube_dim = CubeDim::new_1d(1);

        match nroots {
            1 => rys_root1_kernel::launch::<CpuRuntime>(
                &client, cube_count, cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
                ScalarArg::new(x),
            ).unwrap(),
            2 => rys_root2_kernel::launch::<CpuRuntime>(
                &client, cube_count, cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
                ScalarArg::new(x),
            ).unwrap(),
            3 => rys_root3_kernel::launch::<CpuRuntime>(
                &client, cube_count, cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
                ScalarArg::new(x),
            ).unwrap(),
            4 => rys_root4_kernel::launch::<CpuRuntime>(
                &client, cube_count, cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
                ScalarArg::new(x),
            ).unwrap(),
            5 => rys_root5_kernel::launch::<CpuRuntime>(
                &client, cube_count, cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
                ScalarArg::new(x),
            ).unwrap(),
            _ => panic!("nroots={nroots} not supported in tests"),
        }

        let u_raw = client.read_one(u_handle);
        let w_raw = client.read_one(w_handle);
        let roots = f64::from_bytes(&u_raw)[0..n].to_vec();
        let weights = f64::from_bytes(&w_raw)[0..n].to_vec();
        (roots, weights)
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Test cases
    // ─────────────────────────────────────────────────────────────────────────

    /// Test nroots=1 at small x values (x in [0.1, 2.0]).
    /// Source: rys_roots.c rys_root1() lines 267-328 (x <= 1 and x > 1 segments).
    #[test]
    fn rys_nroots1_small_x() {
        let atol = 1.0e-12_f64;
        for &x in &[0.1_f64, 0.5, 1.0, 2.0] {
            let (roots, weights) = eval_rys_cpu(1, x);
            let (ref_rt, ref_ww) = ref_rys_root1(x);
            assert!(
                (roots[0] - ref_rt).abs() < atol,
                "rys_nroots1_small_x: x={x}: root diff={}",
                (roots[0] - ref_rt).abs()
            );
            assert!(
                (weights[0] - ref_ww).abs() < atol,
                "rys_nroots1_small_x: x={x}: weight diff={}",
                (weights[0] - ref_ww).abs()
            );
        }
    }

    /// Test nroots=1 at large x (asymptotic regime, x > 15).
    /// Source: rys_roots.c rys_root1() lines 271-274 (X > 33 branch).
    #[test]
    fn rys_nroots1_large_x() {
        let atol = 1.0e-12_f64;
        for &x in &[15.0_f64, 30.0, 50.0] {
            let (roots, weights) = eval_rys_cpu(1, x);
            let (ref_rt, ref_ww) = ref_rys_root1(x);
            assert!((roots[0] - ref_rt).abs() < atol,
                "rys_nroots1_large_x: x={x}: root diff={}", (roots[0] - ref_rt).abs());
            assert!((weights[0] - ref_ww).abs() < atol,
                "rys_nroots1_large_x: x={x}: weight diff={}", (weights[0] - ref_ww).abs());
        }
    }

    /// Test nroots=2 across mid-range and asymptotic segments.
    /// Source: rys_roots.c rys_root2() lines 330-487.
    #[test]
    fn rys_nroots2_range() {
        let atol = 1.0e-12_f64;
        for &x in &[1.5_f64, 2.0, 4.0, 40.0, 60.0] {
            let (roots, weights) = eval_rys_cpu(2, x);
            let (ref_roots, ref_weights) = ref_rys_root2(x);
            for i in 0..2 {
                assert!((roots[i] - ref_roots[i]).abs() < atol,
                    "rys_nroots2_range: x={x}: root[{i}] diff={}", (roots[i] - ref_roots[i]).abs());
                assert!((weights[i] - ref_weights[i]).abs() < atol,
                    "rys_nroots2_range: x={x}: weight[{i}] diff={}", (weights[i] - ref_weights[i]).abs());
            }
        }
    }

    /// Test nroots=3 in asymptotic regime.
    /// Source: rys_roots.c rys_root3() last else branch (large-x asymptotic).
    #[test]
    fn rys_nroots3_range() {
        let atol = 1.0e-12_f64;
        for &x in &[10.0_f64, 25.0, 40.0] {
            let (roots, weights) = eval_rys_cpu(3, x);
            let (ref_roots, ref_weights) = ref_rys_root3_asym(x);
            for i in 0..3 {
                assert!((roots[i] - ref_roots[i]).abs() < atol,
                    "rys_nroots3_range: x={x}: root[{i}] diff={}", (roots[i] - ref_roots[i]).abs());
                assert!((weights[i] - ref_weights[i]).abs() < atol,
                    "rys_nroots3_range: x={x}: weight[{i}] diff={}", (weights[i] - ref_weights[i]).abs());
            }
        }
    }

    /// Test nroots=5 in asymptotic regime.
    /// Source: rys_roots.c rys_root5() last else branch (lines ~1604-1616).
    #[test]
    fn rys_nroots5_range() {
        let atol = 1.0e-12_f64;
        for &x in &[10.0_f64, 30.0, 50.0] {
            let (roots, weights) = eval_rys_cpu(5, x);
            let (ref_roots, ref_weights) = ref_rys_root5_asym(x);
            for i in 0..5 {
                assert!((roots[i] - ref_roots[i]).abs() < atol,
                    "rys_nroots5_range: x={x}: root[{i}] diff={}", (roots[i] - ref_roots[i]).abs());
                assert!((weights[i] - ref_weights[i]).abs() < atol,
                    "rys_nroots5_range: x={x}: weight[{i}] diff={}", (weights[i] - ref_weights[i]).abs());
            }
        }
    }

    /// Weight sum identity: sum(w_i) = sqrt(PIE4/x) in the large-x asymptotic regime.
    ///
    /// For large x (>= 40 for nroots=1, >= 33 for nroots>=2 but using >= 50 here to be safe),
    /// all rys_rootN functions use the asymptotic formula where sum(w_i) = sqrt(PIE4/x) exactly.
    /// Polynomial segments do NOT satisfy this identity — they are polynomial approximations
    /// to the Boys function, not to the Gauss-Rys quadrature formula.
    ///
    /// Source: rys_roots.c asymptotic branches (large-x) for all rys_rootN.
    #[test]
    fn rys_weight_sum_identity() {
        let atol = 1.0e-12_f64;
        // Use x >= 50 to ensure ALL nroots variants are in their asymptotic branch.
        // rys_root2 asymptotic threshold is x >= 40; rys_root1 is x > 33.
        for &x in &[50.0_f64, 75.0, 100.0] {
            let f0 = f64::sqrt(PIE4 / x);
            for &nroots in &[1u32, 2, 3, 4, 5] {
                let (_, weights) = eval_rys_cpu(nroots, x);
                let wsum: f64 = weights.iter().sum();
                assert!((wsum - f0).abs() < atol,
                    "rys_weight_sum_identity: nroots={nroots}, x={x}: sum(w)={wsum} vs F_0={f0}, diff={}",
                    (wsum - f0).abs());
            }
        }
    }

    /// Stability test at very small x (1e-10): no NaN or Inf in output.
    /// Source: rys_roots.c small-x branches (x < 3e-7).
    #[test]
    fn rys_small_x_stability() {
        let x = 1.0e-10_f64;
        for &nroots in &[1u32, 2, 3, 4, 5] {
            let (roots, weights) = eval_rys_cpu(nroots, x);
            for (i, &r) in roots.iter().enumerate() {
                assert!(r.is_finite(),
                    "rys_small_x_stability: nroots={nroots}, x=1e-10: root[{i}]={r} not finite");
            }
            for (i, &w) in weights.iter().enumerate() {
                assert!(w.is_finite(),
                    "rys_small_x_stability: nroots={nroots}, x=1e-10: weight[{i}]={w} not finite");
            }
        }
    }

    /// Stability test at large x (45.0): no NaN or Inf in output.
    /// Source: rys_roots.c large-x asymptotic branches.
    #[test]
    fn rys_large_x_stability() {
        let x = 45.0_f64;
        for &nroots in &[1u32, 2, 3, 4, 5] {
            let (roots, weights) = eval_rys_cpu(nroots, x);
            for (i, &r) in roots.iter().enumerate() {
                assert!(r.is_finite(),
                    "rys_large_x_stability: nroots={nroots}, x=45: root[{i}]={r} not finite");
            }
            for (i, &w) in weights.iter().enumerate() {
                assert!(w.is_finite(),
                    "rys_large_x_stability: nroots={nroots}, x=45: weight[{i}]={w} not finite");
            }
        }
    }
}
