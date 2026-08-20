//! Randomized ROCm vendor-parity oracle for the `int1e_grids*` family
//! (originally quick task 260529-twi; the pre-existing eval_raw blocker was
//! fixed in fix/general-contraction-nctr-1e).
//!
//! HISTORY: the CubeCL grids device port (scalar `grids_scalar_kernel` +
//! derivative `grids_deriv_kernel`, dispatched on the ROCm `HipRuntime`)
//! originally could NOT be validated through the usual `eval_raw`-vs-vendor
//! pattern because `eval_raw` rejected the grids 4-element shell tuple
//! (`InvalidShellTuple { expected: 2, got: 4 }`) and never threaded the NGRIDS
//! output axis into the staging/compat sizing. That wiring is now fixed in
//! `cintx-compat::raw` (grids accepts the `[i, j, grid_start, grid_end]` tuple,
//! matching libcint, and folds NGRIDS into the output sizing), so the grids
//! family is exercised end-to-end through `eval_raw` exactly like every other
//! family. The matching CPU vendor-parity gate lives in
//! `unstable_source_parity.rs::grids_parity`.
//!
//! This test drives the grids scalar + all four derivative operators on the
//! ROCm device via the RAW API (`eval_raw`, backend resolved from
//! `CINTX_BACKEND=rocm`) and compares element-wise to vendored libcint 6.1.3.
//! Shell pairs sweep s-s, s-p, p-s, and a NON-SYMMETRIC p-p block (O-2p on two
//! distinct atoms) so both the i↔j AO output layout and the nroots=2 Rys path
//! (which the derivative kernels formerly mis-summed across roots) are covered.
//!
//! Gated `#![cfg(feature = "rocm")]` + `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`
//! runtime gate. The vendor comparison additionally requires
//! `CINTX_ORACLE_BUILD_VENDOR=1` (`#[cfg(has_vendor_libcint)]`). Run via:
//!   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!     cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
//!     --test grids_random_rocm_parity -- --ignored

#![cfg(feature = "rocm")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NGRIDS, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_GRIDS, PTR_ZETA, RawApiId,
    eval_raw,
};

// Deterministic LCG (Numerical Recipes constants) — reproducible random suite
// without an external rng crate (copied from breit_random_rocm_parity.rs).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform f64 in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + frac * (hi - lo)
    }
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Build a 5-shell H2O/STO-3G slab (O-1s, O-2s, O-2p, H1-1s, H2-2p) plus an
/// appended NGRIDS grid block, with per-case jitter: every primitive exponent is
/// scaled by `uniform(0.7, 1.4)`, the two H atom coords are displaced by
/// ±`uniform(0,0.3)` bohr, and the grid points are jittered. Shell 4 is made a
/// p-shell (on H2) so a non-symmetric p-p pair `[2, 4, ..]` (O-2p × H2-2p) is
/// available to exercise the AO i↔j output layout.
fn build_random_h2o_grids(rng: &mut Lcg, ngrids: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let j = |rng: &mut Lcg| rng.uniform(0.7, 1.4);
    let dh = |rng: &mut Lcg| rng.uniform(-0.3, 0.3);

    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307 + dh(rng), 1.1078 + dh(rng)];
    let h2_coord = [0.0_f64, -1.4307 + dh(rng), 1.1078 + dh(rng)];

    let o_1s_exp = [
        130.7093200 * j(rng),
        23.8088610 * j(rng),
        6.4436083 * j(rng),
    ];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513 * j(rng), 1.1695961 * j(rng), 0.3803890 * j(rng)];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513 * j(rng), 1.1695961 * j(rng), 0.3803890 * j(rng)];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509 * j(rng), 0.6239137 * j(rng), 0.1688554 * j(rng)];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let h_2p_exp = [1.1000000 * j(rng), 0.4000000 * j(rng), 0.1500000 * j(rng)];
    let h_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

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
    let h2p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_2p_exp);
    let h2p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_2p_coeff);

    // Appended grid block (jittered around the molecule).
    let ptr_grids_val = env.len() as i32;
    for g in 0..ngrids {
        let t = g as f64 / (ngrids.max(2) - 1) as f64;
        env.extend_from_slice(&[
            -1.0 + 2.0 * t + dh(rng),
            0.5 * (t - 0.5) + dh(rng),
            0.3 + dh(rng),
        ]);
    }
    env[NGRIDS] = ngrids as f64;
    env[PTR_GRIDS] = ptr_grids_val as f64;

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    // (shell, atom, l, exp_ptr, coeff_ptr)
    let shells = [
        (0usize, 0i32, 0i32, o1s_exp_ptr, o1s_coeff_ptr),
        (1, 0, 0, o2s_exp_ptr, o2s_coeff_ptr),
        (2, 0, 1, o2p_exp_ptr, o2p_coeff_ptr),
        (3, 1, 0, h1s_exp_ptr, h1s_coeff_ptr),
        (4, 2, 1, h2p_exp_ptr, h2p_coeff_ptr),
    ];
    let mut bas = vec![0_i32; 5 * BAS_SLOTS];
    for (s, atom, l, ep, cp) in shells {
        bas[s * BAS_SLOTS + ATOM_OF] = atom;
        bas[s * BAS_SLOTS + ANG_OF] = l;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = ep;
        bas[s * BAS_SLOTS + PTR_COEFF] = cp;
    }

    (atm, bas, env)
}

/// Randomized grids ROCm vendor-parity oracle (scalar + 4 derivatives).
#[test]
#[ignore]
fn test_int1e_grids_random_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked with CINTX_ROCM_ORACLE=1 (and CINTX_BACKEND=rocm). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-11_f64;
    let ngrids = 4usize;
    let n_cases = 8usize;
    let mut rng = Lcg::new(0x6a17_d3c0_2605_29b1_u64);

    // (operator symbol, ncomp).
    let ops: [(&str, usize); 5] = [
        ("int1e_grids_sph", 1),
        ("int1e_grids_ip_sph", 3),
        ("int1e_grids_ipvip_sph", 9),
        ("int1e_grids_spvsp_sph", 4),
        ("int1e_grids_ipip_sph", 9),
    ];
    // s-s, s-p, p-s, p-p (non-symmetric: O-2p × H2-2p).
    let pairs: [(i32, i32); 4] = [(0, 1), (0, 2), (2, 3), (2, 4)];

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut total_cases = 0usize;

    for case in 0..n_cases {
        let (atm, bas, env) = build_random_h2o_grids(&mut rng, ngrids);
        let _natm = (atm.len() / ATM_SLOTS) as i32;
        let _nbas = (bas.len() / BAS_SLOTS) as i32;

        for (sym, ncomp) in ops {
            for (si, sj) in pairs {
                let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
                let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
                let n_elem = ncomp * ngrids * ni * nj;
                let shls: [i32; 4] = [si, sj, 0, ngrids as i32];

                // cintx on the ROCm device (backend resolved from CINTX_BACKEND=rocm).
                let mut cintx_out = vec![0.0_f64; n_elem];
                unsafe {
                    eval_raw(
                        RawApiId::Symbol(sym),
                        Some(&mut cintx_out),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!("eval_raw {sym} failed (case {case}, shls {shls:?}): {e:?}")
                    });
                }
                if cintx_out.iter().any(|v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
                total_cases += 1;

                #[cfg(has_vendor_libcint)]
                {
                    use cintx_oracle::vendor_ffi;
                    let mut vendor_out = vec![0.0_f64; n_elem];
                    match sym {
                        "int1e_grids_sph" => vendor_ffi::vendor_int1e_grids_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        ),
                        "int1e_grids_ip_sph" => vendor_ffi::vendor_int1e_grids_ip_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        ),
                        "int1e_grids_ipvip_sph" => vendor_ffi::vendor_int1e_grids_ipvip_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        ),
                        "int1e_grids_spvsp_sph" => vendor_ffi::vendor_int1e_grids_spvsp_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        ),
                        _ => vendor_ffi::vendor_int1e_grids_ipip_sph(
                            &mut vendor_out,
                            &shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        ),
                    };
                    for (idx, (&r, &o)) in vendor_out.iter().zip(cintx_out.iter()).enumerate() {
                        let diff = (o - r).abs();
                        if diff > atol {
                            mismatch_count += 1;
                            if mismatch_count <= 16 {
                                eprintln!(
                                    "  grids rocm MISMATCH case {case} {sym} shls {shls:?} \
                                     idx {idx}: vendor={r:.15e} rocm={o:.15e} diff={diff:.3e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "grids_random_rocm_parity: cases={total_cases} mismatch_count={mismatch_count} \
         any_nonzero={any_nonzero} (atol={atol:.1e})"
    );

    assert!(
        any_nonzero,
        "grids ROCm oracle: all outputs zero across {total_cases} cases — device kernel appears stubbed"
    );

    #[cfg(has_vendor_libcint)]
    assert_eq!(
        mismatch_count, 0,
        "grids ROCm oracle: {mismatch_count} element(s) diverge from vendored libcint 6.1.3 at atol={atol:.1e}"
    );

    #[cfg(not(has_vendor_libcint))]
    eprintln!(
        "WARNING: vendor libcint not built (CINTX_ORACLE_BUILD_VENDOR unset) — \
         ran ROCm device path only, skipped vendor parity comparison."
    );
}
