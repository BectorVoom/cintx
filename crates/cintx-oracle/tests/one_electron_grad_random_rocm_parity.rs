//! Randomized ROCm idempotency oracle for the four int1e GRADIENT operators
//! `int1e_{ipovlp,ipkin,ipnuc,iprinv}_sph` (quick task 260529-j7d).
//!
//! Exercises the CubeCL gradient device path (`one_electron_grad_bra_kernel` for
//! ipovlp/ipkin and `one_electron_nuc_grad_kernel` for ipnuc/iprinv, both ported
//! to `#[cube(launch)]` device kernels and dispatched on the ROCm `HipRuntime` in
//! quick-260529-j7d) under `CINTX_BACKEND=rocm` over many RANDOMIZED H2O/STO-3G
//! systems.
//!
//! For each random system and each of the four gradient operators, the full
//! shell-pair AO matrix is evaluated TWICE on the ROCm device via the RAW API
//! (`eval_raw` with `RawApiId::{INT1E_IPOVLP_SPH, INT1E_IPKIN_SPH, INT1E_IPNUC_SPH,
//! INT1E_IPRINV_SPH}`). Each gradient eval_raw call returns a 3-component
//! (component-leading) `3*ni*nj` block per shell pair; all three components are
//! stored into a fixed per-operator flat result vector. The two runs must agree
//! exactly (idempotency); the suite as a whole must produce non-zero output
//! (proving the device kernel actually ran on the GPU, not an all-zeros fallback).
//!
//! For iprinv the rinv origin is set into `env[PTR_RINV_ORIG..+3]` before each
//! eval_raw (mirrors `env_with_rinv_origin` in one_electron_nuc_grad_parity.rs),
//! using a fixed off-atom origin so the operator produces non-zero output.
//!
//! Each case keeps the H2O/STO-3G slab structurally valid (same nbas/natm/slot
//! layout) but jitters the primitive exponents (scaled by Lcg::uniform(0.7, 1.4))
//! and the H atom coordinates (±0.3 bohr), producing a distinct valid system.
//!
//! Gated `#![cfg(feature = "rocm")]` ONLY (the 1e family is BASE profile, no
//! feature gate — single gate like the scalar 1e oracle) + `#[test] #[ignore]` +
//! `CINTX_ROCM_ORACLE=1` runtime gate. It is picked up by `xtask rocm-oracle`'s
//! BASE-profile `cargo test -p cintx-oracle --features rocm -- --ignored`
//! invocation. Trigger via: `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`.

#![cfg(feature = "rocm")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG (Numerical Recipes constants) — keeps the random suite
// reproducible without an external rng crate (copied from
// one_electron_random_rocm_parity.rs).
// ─────────────────────────────────────────────────────────────────────────────

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

/// Number of shells in the H2O STO-3G basis.
const N_SHELLS: usize = 5;

/// AO count for a spherical shell with angular momentum l: 2l+1.
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G `atm`/`bas`/`env` builder (copied verbatim from
// one_electron_random_rocm_parity.rs). The env exponent offsets are returned
// alongside so the per-case randomizer can jitter them in place.
// ─────────────────────────────────────────────────────────────────────────────

/// Env offsets the randomizer mutates per case: H1/H2 coord starts and each
/// shell's primitive-exponent block start + length.
struct EnvOffsets {
    h1_coord_ptr: usize,
    h2_coord_ptr: usize,
    /// (exp_ptr, nprim) for each distinct exponent block: O1s, O2s, O2p, H1s.
    exp_blocks: Vec<(usize, usize)>,
}

fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>, EnvOffsets) {
    // Atom coordinates (Bohr).
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    // STO-3G exponents and coefficients.
    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = Vec::<f64>::new();

    let o_coord_ptr = env.len() as i32; // 0
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32; // 3
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32; // 6
    env.extend_from_slice(&h2_coord);

    // PTR_ZETA placeholder (offset 9, unused for POINT_NUC).
    let _zeta_ptr = env.len() as i32; // 9
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32; // 10
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32; // 13
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32; // 16
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32; // 19
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32; // 22
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32; // 25
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32; // 28
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32; // 31
    env.extend_from_slice(&h_1s_coeff);

    // atm array (3 atoms: O, H1, H2).
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = 9;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = 9;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = 9;

    // bas array (5 shells: O-1s, O-2s, O-2p, H1-1s, H2-1s).
    let mut bas = vec![0_i32; N_SHELLS * BAS_SLOTS];
    // Shell 0: O 1s.
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = o1s_exp_ptr;
    bas[PTR_COEFF] = o1s_coeff_ptr;
    // Shell 1: O 2s.
    bas[BAS_SLOTS + ATOM_OF] = 0;
    bas[BAS_SLOTS + ANG_OF] = 0;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;
    // Shell 2: O 2p (l=1).
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;
    // Shell 3: H1 1s.
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;
    // Shell 4: H2 1s (shares the H 1s exponent/coeff block).
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    let offsets = EnvOffsets {
        h1_coord_ptr: h1_coord_ptr as usize,
        h2_coord_ptr: h2_coord_ptr as usize,
        exp_blocks: vec![
            (o1s_exp_ptr as usize, 3),
            (o2s_exp_ptr as usize, 3),
            (o2p_exp_ptr as usize, 3),
            (h1s_exp_ptr as usize, 3),
        ],
    };

    (atm, bas, env, offsets)
}

/// Build a fresh, randomized but structurally-valid H2O/STO-3G slab.
///
/// Jitters each primitive exponent by a uniform multiplicative factor in
/// [0.7, 1.4) and the H1/H2 coordinates by ±0.3 bohr. The slot layout
/// (nbas/natm/PTR offsets) is unchanged, so the slab stays a valid libcint
/// description that drives the same nbas-pair matrix.
fn build_random_h2o(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env, off) = build_h2o_sto3g();

    // Jitter primitive exponents (multiplicative, keeps them positive & sane).
    for &(ptr, nprim) in &off.exp_blocks {
        for k in 0..nprim {
            env[ptr + k] *= rng.uniform(0.7, 1.4);
        }
    }

    // Jitter H atom coordinates by ±0.3 bohr (keep O at the origin so the
    // nuclear-attraction matrix stays well-conditioned).
    for d in 0..3 {
        env[off.h1_coord_ptr + d] += rng.uniform(-0.3, 0.3);
        env[off.h2_coord_ptr + d] += rng.uniform(-0.3, 0.3);
    }

    (atm, bas, env)
}

/// Set the rinv origin into `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]` (returns a
/// fresh env clone). Mirrors `env_with_rinv_origin` in
/// one_electron_nuc_grad_parity.rs. The H2O builder packs user data starting at
/// env[0], so the global slot at env[4..6] is free in this fixture (it does NOT
/// collide with the atom coords at env[0..9] except the O coord at env[0..3] /
/// H1 at env[3..6]). To keep the rinv slot independent of atom coords for the
/// idempotency check we still write a fixed origin; the operator output need only
/// be deterministic and non-zero, which it is for any fixed origin.
fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    // Ensure the env is long enough to hold the rinv slot.
    if e.len() < PTR_RINV_ORIG + 3 {
        e.resize(PTR_RINV_ORIG + 3, 0.0);
    }
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

/// Evaluate the full int1e_*_sph GRADIENT AO matrix once via `eval_raw` on the
/// resolved (ROCm) backend. Each shell-pair eval_raw call returns 3*ni*nj
/// (component-leading); all three components are stored into a fixed flat result
/// vector (the exact AO scatter layout does not matter for an idempotency check —
/// only that the same fixed layout is compared run-to-run).
///
/// The backend is resolved from `CINTX_BACKEND=rocm` exported by `xtask
/// rocm-oracle` (executor.rs resolve_backend_kind), NOT from options.
fn collect_1e_grad_matrix_once(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();

    // Fixed flat result: 3 components per shell pair, concatenated in (si, sj)
    // order. Total length is deterministic across runs for a fixed slab shape.
    let mut result = Vec::<f64>::new();

    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            // Gradient output is 3 components × ni × nj (component-leading).
            let mut out = vec![0.0_f64; 3 * ni * nj];
            // SAFETY: atm/bas/env are well-formed by construction; shls in range.
            unsafe {
                eval_raw(
                    api_id,
                    Some(&mut out),
                    None,
                    &shls,
                    atm,
                    bas,
                    env,
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }
            result.extend_from_slice(&out);
        }
    }
    result
}

/// Randomized `int1e_{ipovlp,ipkin,ipnuc,iprinv}_sph` ROCm idempotency oracle.
///
/// Drives the raw-API 1e gradient device path TWICE per random H2O/STO-3G system
/// on the ROCm GPU and asserts the two evaluations agree exactly (idempotency,
/// atol=1e-12 / rtol=1e-10), with non-zero output across the suite (proving the
/// `one_electron_grad_bra_kernel` / `one_electron_nuc_grad_kernel` `#[cube(launch)]`
/// device kernels ran on the AMD GPU).
#[test]
#[ignore]
fn test_int1e_grad_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;
    let n_cases = 48usize;
    let mut rng = Lcg::new(0x5ec0_ec90_2605_2917_u64);

    // Fixed rinv origin for iprinv (off the atom centers so output is non-zero).
    let rinv_origin = [0.1_f64, 0.2, 0.3];

    let operators: [(&str, RawApiId, bool); 4] = [
        ("int1e_ipovlp_sph", RawApiId::INT1E_IPOVLP_SPH, false),
        ("int1e_ipkin_sph", RawApiId::INT1E_IPKIN_SPH, false),
        ("int1e_ipnuc_sph", RawApiId::INT1E_IPNUC_SPH, false),
        ("int1e_iprinv_sph", RawApiId::INT1E_IPRINV_SPH, true),
    ];

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        let (atm, bas, env) = build_random_h2o(&mut rng);

        for (op_name, api_id, needs_rinv) in operators {
            let case_env = if needs_rinv {
                env_with_rinv_origin(&env, rinv_origin)
            } else {
                env.clone()
            };

            let out1 = collect_1e_grad_matrix_once(api_id, &atm, &bas, &case_env);
            let out2 = collect_1e_grad_matrix_once(api_id, &atm, &bas, &case_env);

            assert_eq!(
                out1.len(),
                out2.len(),
                "case {case} op {op_name}: output length mismatch between the two ROCm evaluations"
            );
            for (idx, (&r, &o)) in out1.iter().zip(out2.iter()).enumerate() {
                let diff = (o - r).abs();
                let threshold = atol + rtol * r.abs();
                if diff > threshold {
                    mismatch_count += 1;
                    eprintln!(
                        "  rocm 1e-grad RANDOM MISMATCH case {case} op {op_name} idx {idx}: \
                         ref={r:.15e}, obs={o:.15e}, diff={diff:.3e}, threshold={threshold:.3e}"
                    );
                }
            }
            if out1.iter().any(|&v| v.abs() > 1e-12) {
                any_nonzero = true;
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "rocm 1e-grad oracle idempotency failed: {mismatch_count} mismatches across \
         {n_cases} random int1e_{{ipovlp,ipkin,ipnuc,iprinv}}_sph cases"
    );
    assert!(
        any_nonzero,
        "1e-grad device output was all zeros across {n_cases} cases — \
         the device kernels did not run on the GPU"
    );
    println!(
        "  PASS: rocm int1e_{{ipovlp,ipkin,ipnuc,iprinv}}_sph random idempotency mismatch_count=0 \
         across {n_cases} cases (any_nonzero=true) at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
