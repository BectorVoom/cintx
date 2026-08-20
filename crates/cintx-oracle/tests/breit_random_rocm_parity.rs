//! Randomized ROCm vendor-parity oracle for `int2e_breit_{r1p2,r2p2}_spinor`
//! (quick task 260529-twi).
//!
//! Exercises the CubeCL `breit_g_kernel` GPU device path — the per-quartet Breit
//! 4-center G-tensor BUILD (VRR + the ibase/kbase-selected 4-branch HRR), ported
//! to a `#[cube(launch)]` device kernel generic over `F` and dispatched on the
//! ROCm `HipRuntime` in quick-260529-twi. The host-side gout_breit operator
//! ladder (`gout_breit_r1p2`/`_r2p2` + `nabla1{i,j,l}` + `x1{j,l}`) and the
//! `cart_to_spinor_sf_4d` transform stay on host (documented split), so this
//! oracle exercises the device G-tensor feeding the unchanged host spinor tail.
//!
//! For each randomized H2O/STO-3G system (primitive exponents scaled by
//! `Lcg::uniform(0.7, 1.4)`, H atom coords jittered ±0.3 bohr) and each of the
//! two Breit spinor operators, the spinor integral is evaluated on the ROCm
//! device via the RAW API (`eval_raw(RawApiId::Symbol("int2e_breit_*_spinor"))`,
//! backend resolved from `CINTX_BACKEND=rocm`) and compared element-wise to the
//! vendored libcint 6.1.3 reference (`vendor_ffi::vendor_int2e_breit_*_spinor`)
//! at atol=1e-12. Two shell quartets are swept: the square O-1s/O-2s pair
//! `[0,1,0,1]` and a NON-SQUARE quartet `[2,3,0,1]` (O-2p bra) so a spinor
//! KET↔BRA transpose error cannot hide behind a symmetric block.
//!
//! Gated `#![cfg(feature = "rocm")]` + `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`
//! runtime gate. The vendor comparison additionally requires
//! `CINTX_ORACLE_BUILD_VENDOR=1` (`#[cfg(has_vendor_libcint)]`). Run via:
//!   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!     cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
//!     --test breit_random_rocm_parity -- --ignored

#![cfg(feature = "rocm")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

// Deterministic LCG (Numerical Recipes constants) — reproducible random suite
// without an external rng crate (copied from one_electron_random_rocm_parity.rs).
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

/// Build the H2O/STO-3G `atm`/`bas`/`env` slab (5 shells: O-1s, O-2s, O-2p, H1-1s,
/// H2-1s), with per-case jitter: every primitive exponent is scaled by
/// `uniform(0.7, 1.4)` and the two H atom coords are displaced by ±`uniform(0,0.3)`
/// bohr. Structurally identical to `unstable_source_parity.rs::build_h2o_sto3g_breit`.
fn build_random_h2o_breit(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = o1s_exp_ptr;
    bas[PTR_COEFF] = o1s_coeff_ptr;

    bas[BAS_SLOTS + ATOM_OF] = 0;
    bas[BAS_SLOTS + ANG_OF] = 0;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

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

/// Spinor output size (f64 count) for a kappa=0 shell quartet: 2*∏ 2*(2l+1)
/// (per-shell CINTcgto_spinor for kappa=0 is 2*(2l+1); the trailing factor 2 is
/// the complex re/im interleave). NB: 2*(2l+1) == 2*(l+1) only for l=0, so an
/// l>0 (e.g. O-2p) shell in the quartet REQUIRES this form.
fn spinor_n_elem_kappa0(bas: &[i32], shls: &[i32; 4]) -> usize {
    let mut ns = 1usize;
    for &sh in shls {
        let l = bas[sh as usize * BAS_SLOTS + ANG_OF] as usize;
        ns *= 2 * (2 * l + 1);
    }
    2 * ns
}

/// Randomized Breit spinor ROCm vendor-parity oracle.
#[test]
#[ignore]
fn test_int2e_breit_spinor_random_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked with CINTX_ROCM_ORACLE=1 (and CINTX_BACKEND=rocm). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let n_cases = 12usize;
    let mut rng = Lcg::new(0x5ec0_ec90_2605_2bb1_u64);

    // (operator symbol, vendor fn selector). Square [0,1,0,1] + non-square [2,3,0,1].
    let symbols = ["int2e_breit_r1p2_spinor", "int2e_breit_r2p2_spinor"];
    let quartets: [[i32; 4]; 2] = [[0, 1, 0, 1], [2, 3, 0, 1]];

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut total_cases = 0usize;

    for case in 0..n_cases {
        let (atm, bas, env) = build_random_h2o_breit(&mut rng);
        let _natm = (atm.len() / ATM_SLOTS) as i32;
        let _nbas = (bas.len() / BAS_SLOTS) as i32;

        for sym in symbols {
            for shls in &quartets {
                let n_elem = spinor_n_elem_kappa0(&bas, shls);

                // cintx on the ROCm device (backend resolved from CINTX_BACKEND=rocm).
                let mut cintx_out = vec![0.0_f64; n_elem];
                unsafe {
                    eval_raw(
                        RawApiId::Symbol(sym),
                        Some(&mut cintx_out),
                        None,
                        shls,
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
                    if sym == "int2e_breit_r1p2_spinor" {
                        vendor_ffi::vendor_int2e_breit_r1p2_spinor(
                            &mut vendor_out,
                            shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        );
                    } else {
                        vendor_ffi::vendor_int2e_breit_r2p2_spinor(
                            &mut vendor_out,
                            shls,
                            &atm,
                            _natm,
                            &bas,
                            _nbas,
                            &env,
                        );
                    }
                    for (idx, (&r, &o)) in vendor_out.iter().zip(cintx_out.iter()).enumerate() {
                        let diff = (o - r).abs();
                        if diff > atol {
                            mismatch_count += 1;
                            if mismatch_count <= 16 {
                                eprintln!(
                                    "  breit rocm MISMATCH case {case} {sym} shls {shls:?} \
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
        "breit_random_rocm_parity: cases={total_cases} mismatch_count={mismatch_count} \
         any_nonzero={any_nonzero} (atol={atol:.1e})"
    );

    assert!(
        any_nonzero,
        "breit ROCm oracle: all outputs zero across {total_cases} cases — device kernel appears stubbed"
    );

    #[cfg(has_vendor_libcint)]
    assert_eq!(
        mismatch_count, 0,
        "breit ROCm oracle: {mismatch_count} element(s) diverge from vendored libcint 6.1.3 at atol={atol:.1e}"
    );

    #[cfg(not(has_vendor_libcint))]
    eprintln!(
        "WARNING: vendor libcint not built (CINTX_ORACLE_BUILD_VENDOR unset) — \
         ran ROCm device path only, skipped vendor parity comparison."
    );
}
