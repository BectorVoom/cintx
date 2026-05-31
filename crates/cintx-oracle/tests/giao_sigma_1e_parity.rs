//! Phase 30 (GIAO-03) — GIAO×σ 1e spinor vendor byte-identity parity.
//!
//! Wave 0 (this plan, 30-00): the **gauge-gout byte-identity micro-test**
//! (`giao_sigma_micro`) — the D-03 de-risk gate. It proves the genuinely-new
//! device math of Phase 30 (the gauge-origin `x1i`-with-origin fold in
//! `sigma_p.rs`) byte-identical to vendored libcint's `int1e_cg_sa10sp_spinor`
//! at atol=1e-12, on a combined gauge≠0 ∧ kappa≠0 fixture, PLUS a
//! cg→giao-at-origin=0 collapse differential check proving the gauge term is
//! live (not silently zeroed).
//!
//! Waves 1 (Plan 30-01) extends this same file with the full 9-family 1e parity
//! scaffold (collectors + per-family byte-identity gate macro + `test_no_silent_skip`).
//! Those are left `#[ignore]`-stubbed RED below so Wave 1 reuses this file.
//!
//! Double-gate (memory `reference_oracle_vendor_parity_invocation`): real vendor
//! byte-identity needs BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`.
//! Without both, the vendor bodies compile out and only the always-on sizing
//! guard runs:
//!   `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_micro`

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COMMON_ORIG,
    PTR_COORD, PTR_EXP,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

/// The nine 1e GIAO×σ family symbols this scaffold gates (GIAO-03). Wave 1 wires
/// all nine; Wave 0 proves only `int1e_cg_sa10sp_spinor` (+ the giao collapse).
#[allow(dead_code)]
const GIAO_1E_FAMILIES: &[&str] = &[
    "int1e_spgsp_spinor",
    "int1e_spgnucsp_spinor",
    "int1e_spgsa01_spinor",
    "int1e_cg_sa10sp_spinor",
    "int1e_cg_sa10nucsp_spinor",
    "int1e_cg_sa10sa01_spinor",
    "int1e_giao_sa10sp_spinor",
    "int1e_giao_sa10nucsp_spinor",
    "int1e_giao_sa10sa01_spinor",
];

/// Per-family component_rank (output spinor-matrix count). The `sp`/`nucsp` arms
/// are rank 3 (ng[7]=3); the `sa01` arms are rank 9 (ng[7]=9). Wave 0 only uses
/// the rank-3 `cg_sa10sp` / `giao_sa10sp` arms.
#[allow(dead_code)]
fn family_component_rank(family: &str) -> usize {
    if family.ends_with("sa01_spinor") {
        9
    } else {
        3
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// spinor_len — kappa≠0 GT/LT sizing (libcint _len_spinor, cart2sph.c:3537).
//   kappa == 0 → 4l+2  (both blocks)
//   kappa  < 0 → 2l+2  (GT, j = l+1/2)
//   kappa  > 0 → 2l    (LT, j = l−1/2)
// ─────────────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
fn spinor_len(l: i32, kappa: i32) -> usize {
    if kappa == 0 {
        (4 * l + 2) as usize
    } else if kappa < 0 {
        (2 * l + 2) as usize
    } else {
        (2 * l) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract a single shell's primitive data from atm/bas/env (env coeff is
// COLUMN-major [ic*nprim+ip]; transposed to ROW-major on extraction —
// project_raw_nctr_coeff_transpose).
// ─────────────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
struct ShellData {
    l: u8,
    kappa: i16,
    nprim: usize,
    nctr: usize,
    coord: [f64; 3],
    exps: Vec<f64>,
    coeff_row_major: Vec<f64>,
}

#[allow(dead_code)]
fn extract_shell(s: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> ShellData {
    let l = bas[s * BAS_SLOTS + ANG_OF] as u8;
    let kappa = bas[s * BAS_SLOTS + KAPPA_OF] as i16;
    let nprim = bas[s * BAS_SLOTS + NPRIM_OF] as usize;
    let nctr = bas[s * BAS_SLOTS + NCTR_OF] as usize;
    let atom = bas[s * BAS_SLOTS + ATOM_OF] as usize;
    let exp_ptr = bas[s * BAS_SLOTS + PTR_EXP] as usize;
    let coeff_ptr = bas[s * BAS_SLOTS + PTR_COEFF] as usize;
    let coord_ptr = atm[atom * ATM_SLOTS + PTR_COORD] as usize;

    let coord = [env[coord_ptr], env[coord_ptr + 1], env[coord_ptr + 2]];
    let exps = env[exp_ptr..exp_ptr + nprim].to_vec();

    let col_major = &env[coeff_ptr..coeff_ptr + nprim * nctr];
    let mut coeff_row_major = vec![0.0_f64; nprim * nctr];
    for ic in 0..nctr {
        for ip in 0..nprim {
            coeff_row_major[ip * nctr + ic] = col_major[ic * nprim + ip];
        }
    }

    ShellData { l, kappa, nprim, nctr, coord, exps, coeff_row_major }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity helpers (cloned from rel_1e_sigma_parity.rs).
// ─────────────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            if mismatches <= 20 {
                eprintln!(
                    "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                     diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
    }
    mismatches
}

#[allow(dead_code)]
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(
        any_nonzero,
        "{label}: spinor matrix is all-zero (zero-fill / silent-skip regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collector — drive the Phase-30 gauge launcher
// (kernels::sigma_p::launch_int1e_cg_sa10sp_spinor_pair) for the shell pair
// (0, 1). `common_orig` is read from env[PTR_COMMON_ORIG..+3]; the launcher
// receives dri = ri − common_orig. Returns the flat interleaved-complex rank-3
// spinor block: out[grp*(ni_sp*nj_sp*2) + (j*ni_sp + i)*2 + {0:re,1:im}].
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_cg_sa10sp(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::launch_int1e_cg_sa10sp_spinor_pair;
    use cintx_runtime::{BackendIntent, BackendKind};

    let si = extract_shell(0, atm, bas, env);
    let sj = extract_shell(1, atm, bas, env);

    let di = spinor_len(si.l as i32, si.kappa as i32);
    let dj = spinor_len(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend must initialise");

    // dri = ri − common_orig (bra-side gauge fold for the cg_sa10* arms).
    let common_orig = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    let dri = [
        si.coord[0] - common_orig[0],
        si.coord[1] - common_orig[1],
        si.coord[2] - common_orig[2],
    ];

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * 2 * 3];
    launch_int1e_cg_sa10sp_spinor_pair::<f64>(
        &backend,
        si.l,
        si.kappa,
        sj.l,
        sj.kappa,
        si.nprim,
        sj.nprim,
        si.nctr,
        sj.nctr,
        si.coord,
        sj.coord,
        dri,
        &si.exps,
        &sj.exps,
        &si.coeff_row_major,
        &sj.coeff_row_major,
        &mut staging,
    )
    .expect("int1e_cg_sa10sp spinor launch must succeed");
    staging
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collector — REAL FFI shim. Single shell pair (0, 1); out sized
// 3*ni_sp*nj_sp*2 (rank 3, interleaved re/im, kappa≠0 non-(4l+2) sizing).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
fn collect_vendor_giao_1e(family: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int1e_cg_sa10sp_spinor, vendor_int1e_giao_sa10sp_spinor,
    };

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2 * family_component_rank(family)];
    match family {
        "int1e_cg_sa10sp_spinor" => {
            vendor_int1e_cg_sa10sp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_giao_sa10sp_spinor" => {
            vendor_int1e_giao_sa10sp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        other => panic!("Wave-0 micro-test only drives cg/giao_sa10sp; got {other}"),
    };
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Always-on kappa-sizing guard (Pitfall 4): the combined fixture MUST ride the
// non-(4l+2) GT/LT sizing path. Runs even on a determinism-only (non-vendor)
// build so a sizing regression is caught regardless of gate flags.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_sizing_non_4l_plus_2() {
    assert_eq!(spinor_len(1, 1), 2, "p kappa=+1 (LT) → di = 2*1 = 2");
    assert_eq!(spinor_len(2, -1), 6, "d kappa=−1 (GT) → dj = 2*2+2 = 6");
    assert_eq!(spinor_len(1, 0), 6, "p kappa=0 would be 4l+2 = 6 (NOT used)");
    assert_eq!(spinor_len(2, 0), 10, "d kappa=0 would be 4l+2 = 10 (NOT used)");

    let (_atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    assert_eq!(bas[ANG_OF], 1, "shell 0 = p");
    assert_eq!(bas[KAPPA_OF], 1, "shell 0 kappa = +1 (LT)");
    assert_eq!(bas[NCTR_OF], 2, "shell 0 nctr = 2 (general contraction)");
    assert_eq!(bas[BAS_SLOTS + ANG_OF], 2, "shell 1 = d");
    assert_eq!(bas[BAS_SLOTS + KAPPA_OF], -1, "shell 1 kappa = −1 (GT)");
    // D-02 constraint (1): the gauge origin is genuinely non-zero.
    let origin = [env[PTR_COMMON_ORIG], env[PTR_COMMON_ORIG + 1], env[PTR_COMMON_ORIG + 2]];
    assert!(
        origin.iter().any(|&c| c != 0.0),
        "build_gauge_kappa_spinor_fixture must set a non-zero gauge origin"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Wave-0 gauge-gout byte-identity micro-test (D-03 de-risk gate).
//
//   1. Build the combined gauge≠0 ∧ kappa≠0 fixture.
//   2. Drive int1e_cg_sa10sp_spinor through the extended sigma_p.rs gauge variant.
//   3. Assert both arms non-zero AND count_mismatches(vendor, cintx, ATOL, RTOL)==0.
//   4. DIFFERENTIAL collapse: recompute with common_orig=[0,0,0]. Because the
//      fixture's bra (i) shell sits at the coordinate origin, dri = ri − 0 = 0,
//      so the gauge x1i step collapses to G1E_R_I — the int1e_giao_sa10sp build.
//      Assert it byte-equals vendor int1e_giao_sa10sp_spinor. This is the
//      independent witness that the origin plumbing is live, not silently zeroed.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_micro() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();

    // Sanity: the Wave-0 fixture's bra (i) shell is at the coordinate origin, so
    // the common_orig=[0,0,0] collapse arm gives dri=0 → cg must equal giao.
    let si = extract_shell(0, &atm, &bas, &env);
    assert_eq!(si.coord, [0.0, 0.0, 0.0], "Wave-0 collapse requires bra at origin");

    // ── (A) Byte-identity vs vendor at the genuine non-zero gauge origin. ──
    let vendor_cg = collect_vendor_giao_1e("int1e_cg_sa10sp_spinor", &atm, &bas, &env);
    let cintx_cg = collect_cintx_cg_sa10sp(&atm, &bas, &env);
    assert_eq!(cintx_cg.len(), vendor_cg.len(), "cg_sa10sp length");
    assert_any_nonzero(&cintx_cg, "int1e_cg_sa10sp cintx");
    assert_any_nonzero(&vendor_cg, "int1e_cg_sa10sp vendor");
    assert_eq!(
        count_mismatches(&vendor_cg, &cintx_cg, ATOL, RTOL),
        0,
        "int1e_cg_sa10sp: mismatches vs vendored libcint at atol={ATOL}"
    );

    // ── (B) cg→giao collapse at common_orig=[0,0,0] (dri=0 ⇒ G1E_R_I). ──
    let mut env0 = env.clone();
    env0[PTR_COMMON_ORIG] = 0.0;
    env0[PTR_COMMON_ORIG + 1] = 0.0;
    env0[PTR_COMMON_ORIG + 2] = 0.0;
    let cintx_collapse = collect_cintx_cg_sa10sp(&atm, &bas, &env0);
    let vendor_giao = collect_vendor_giao_1e("int1e_giao_sa10sp_spinor", &atm, &bas, &env);
    assert_any_nonzero(&cintx_collapse, "cg→giao collapse cintx");
    assert_any_nonzero(&vendor_giao, "int1e_giao_sa10sp vendor");
    assert_eq!(
        count_mismatches(&vendor_giao, &cintx_collapse, ATOL, RTOL),
        0,
        "cg_sa10sp at common_orig=[0,0,0] MUST collapse to int1e_giao_sa10sp \
         (proves the gauge term is live, not silently zeroed)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Wave 1 (Plan 30-01) RED stubs — the full 9-family 1e parity gate + no-silent-skip
// integrity assertion extend THIS file. Left #[ignore]d so Wave 1 reuses the
// scaffold above (GIAO_1E_FAMILIES, the collectors, count_mismatches).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
#[ignore = "Wave 1 (Plan 30-01): wire all 9 GIAO×σ 1e families + no-silent-skip"]
fn giao_sigma_1e_full_parity_red() {
    // Wave 1 will iterate GIAO_1E_FAMILIES, drive collect_cintx_giao_1e /
    // collect_vendor_giao_1e per family on build_gauge_kappa_spinor_fixture, and
    // assert count_mismatches(..., ATOL, RTOL)==0 + MANIFEST oracle_covered=true.
    let _ = GIAO_1E_FAMILIES;
    let _ = family_component_rank;
    unimplemented!("Wave 1 wires the remaining 8 families onto the proven fold");
}
