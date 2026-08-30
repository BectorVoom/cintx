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
//! Wave 1 (Plans 30-01a/b/c/d) extends this same file with the full 9-family 1e
//! parity scaffold (collectors + per-family byte-identity gates + the full-set
//! `test_no_silent_skip`). As of 30-01d all nine families are live and green (no
//! ignored gates) — the 1e half of GIAO-03 is closed.
//!
//! Double-gate (memory `reference_oracle_vendor_parity_invocation`): real vendor
//! byte-identity needs BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`.
//! Without both, the vendor bodies compile out and only the always-on sizing
//! guard runs:
//!   `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_micro`

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COMMON_ORIG,
    PTR_COORD, PTR_EXP, PTR_RINV_ORIG,
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

    ShellData {
        l,
        kappa,
        nprim,
        nctr,
        coord,
        exps,
        coeff_row_major,
    }
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
    let common_orig = [
        env[PTR_COMMON_ORIG],
        env[PTR_COMMON_ORIG + 1],
        env[PTR_COMMON_ORIG + 2],
    ];
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
    assert_eq!(
        spinor_len(1, 0),
        6,
        "p kappa=0 would be 4l+2 = 6 (NOT used)"
    );
    assert_eq!(
        spinor_len(2, 0),
        10,
        "d kappa=0 would be 4l+2 = 10 (NOT used)"
    );

    let (_atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    assert_eq!(bas[ANG_OF], 1, "shell 0 = p");
    assert_eq!(bas[KAPPA_OF], 1, "shell 0 kappa = +1 (LT)");
    assert_eq!(bas[NCTR_OF], 2, "shell 0 nctr = 2 (general contraction)");
    assert_eq!(bas[BAS_SLOTS + ANG_OF], 2, "shell 1 = d");
    assert_eq!(bas[BAS_SLOTS + KAPPA_OF], -1, "shell 1 kappa = −1 (GT)");
    // D-02 constraint (1): the gauge origin is genuinely non-zero.
    let origin = [
        env[PTR_COMMON_ORIG],
        env[PTR_COMMON_ORIG + 1],
        env[PTR_COMMON_ORIG + 2],
    ];
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
    assert_eq!(
        si.coord,
        [0.0, 0.0, 0.0],
        "Wave-0 collapse requires bra at origin"
    );

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

// ═════════════════════════════════════════════════════════════════════════════
// Sub-wave 1c (Plan 30-01c): the rank-9 Rys+gauge sa01 byte-identity gates.
//
// int1e_cg_sa10sa01 / int1e_giao_sa10sa01 — component_rank 9, REAL c2s_si_1e.
// Drives the rank-9 dispatcher (kernels::sigma_1e::launch_int1e_giao_sigma_family_
// spinor_pair) on the NON-SQUARE p×d combined gauge∧kappa fixture; asserts
// byte-identity vs vendor at atol=1e-12 across ALL 9 components + all-9-non-zero
// (Pitfall 3 truncation guard) + the cg→giao collapse witness.
// ═════════════════════════════════════════════════════════════════════════════

/// Drive the rank-9 sa01 dispatcher for the shell pair (0,1). `op` is
/// `cg_sa10sa01` or `giao_sa10sa01`. Returns the flat interleaved-complex rank-9
/// spinor block (9*ni_sp*nj_sp*2). common_orig is read from env (the dispatcher
/// forms dri=ri−common_orig for cg; giao ignores it, using origin=ri internally).
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_sa10sa01(op: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_runtime::{BackendIntent, BackendKind};

    let si = extract_shell(0, atm, bas, env);
    let sj = extract_shell(1, atm, bas, env);

    let di = spinor_len(si.l as i32, si.kappa as i32);
    let dj = spinor_len(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;
    // NON-SQUARE block (Pitfall 6 / T-30-01c-06): p(LT,nctr=2)×d(GT).
    assert_ne!(ni_sp, nj_sp, "sa01 gate requires a NON-SQUARE spinor block");

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend must initialise");

    let common_orig = [
        env[PTR_COMMON_ORIG],
        env[PTR_COMMON_ORIG + 1],
        env[PTR_COMMON_ORIG + 2],
    ];
    // 30-01c root cause: sa01 evaluates at the SINGLE rinv center env[PTR_RINV_ORIG]
    // (intor3.c int1e_type=1, charge +1) — must be threaded, not assumed [0,0,0].
    let rinv_orig = [
        env[PTR_RINV_ORIG],
        env[PTR_RINV_ORIG + 1],
        env[PTR_RINV_ORIG + 2],
    ];

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * 2 * 9];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend,
        op,
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
        common_orig,
        rinv_orig,
        &si.exps,
        &sj.exps,
        &si.coeff_row_major,
        &sj.coeff_row_major,
        // sa01 is rinv (single center, threaded via rinv_orig) → no nuclear centers.
        &[],
        &[],
        &mut staging,
    )
    .unwrap_or_else(|e| panic!("{op} rank-9 spinor launch must succeed: {e:?}"));
    staging
}

/// Vendor rank-9 sa01 collector (REAL FFI shim). out sized 9*ni_sp*nj_sp*2.
#[cfg(has_vendor_libcint)]
fn collect_vendor_sa10sa01(family: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int1e_cg_sa10sa01_spinor, vendor_int1e_giao_sa10sa01_spinor,
    };

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    // family_component_rank returns 9 for *_sa01_spinor → buffer sized for all 9.
    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2 * family_component_rank(family)];
    match family {
        "int1e_cg_sa10sa01_spinor" => {
            vendor_int1e_cg_sa10sa01_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_giao_sa10sa01_spinor" => {
            vendor_int1e_giao_sa10sa01_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        other => panic!("Sub-wave 1c only drives cg/giao_sa10sa01; got {other}"),
    };
    out
}

/// Assert that ALL `rank` component slices have at least one non-zero element
/// (Pitfall 3: a rank-3 truncation would leave 6 of the 9 slices all-zero).
#[cfg(has_vendor_libcint)]
fn assert_all_components_nonzero(matrix: &[f64], rank: usize, label: &str) {
    assert_eq!(
        matrix.len() % rank,
        0,
        "{label}: length not divisible by rank {rank}"
    );
    let comp_len = matrix.len() / rank;
    for c in 0..rank {
        let slice = &matrix[c * comp_len..(c + 1) * comp_len];
        assert!(
            slice.iter().any(|v| v.abs() > 1e-14),
            "{label}: component {c} of {rank} is all-zero (rank-truncation regression)"
        );
    }
}

/// cg_sa10sa01 byte-identity vs vendor at atol=1e-12 on the non-square block.
///
/// GREEN (30-01c root cause fixed): the SUMMARY/RESEARCH "gout→gc-block layout"
/// hypothesis was wrong. The cart-path discriminator (sa01_cart_discriminator.rs)
/// proved the g-tensor AND the c2s_si_1e transform both byte-correct. The real bug
/// was the `cg/giao_sa10sa01` dispatch arms hardcoding the rinv center as [0,0,0]
/// instead of `env[PTR_RINV_ORIG]` (now threaded via `rinv_orig`). The mismatch only
/// surfaced on the non-zero-rinv fixture; controlled fixtures with PTR_RINV_ORIG=0
/// matched all along.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_cg_sa10sa01() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_sa10sa01("int1e_cg_sa10sa01_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_sa10sa01("cg_sa10sa01", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "cg_sa10sa01 length (rank 9)");
    assert_all_components_nonzero(&cintx, 9, "cg_sa10sa01 cintx");
    assert_all_components_nonzero(&vendor, 9, "cg_sa10sa01 vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_cg_sa10sa01: mismatches vs vendored libcint at atol={ATOL} (rank 9)"
    );
}

/// giao_sa10sa01 byte-identity vs vendor at atol=1e-12 on the non-square block.
/// GREEN (30-01c root cause fixed) — see `giao_sigma_1e_cg_sa10sa01` (shared rinv fix).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_giao_sa10sa01() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_sa10sa01("int1e_giao_sa10sa01_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_sa10sa01("giao_sa10sa01", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "giao_sa10sa01 length (rank 9)");
    assert_all_components_nonzero(&cintx, 9, "giao_sa10sa01 cintx");
    assert_all_components_nonzero(&vendor, 9, "giao_sa10sa01 vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_giao_sa10sa01: mismatches vs vendored libcint at atol={ATOL} (rank 9)"
    );
}

/// cg→giao differential witness: at common_orig=[0,0,0] with the bra (i) shell at
/// the coordinate origin, dri = ri − 0 = ri = the giao natural-center origin, so
/// cg_sa10sa01 MUST collapse to vendor int1e_giao_sa10sa01 — proving the rank-9
/// gauge term is live, not silently zeroed.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_sa10sa01_cg_giao_collapse() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let si = extract_shell(0, &atm, &bas, &env);
    assert_eq!(
        si.coord,
        [0.0, 0.0, 0.0],
        "collapse witness requires bra at origin"
    );

    let mut env0 = env.clone();
    env0[PTR_COMMON_ORIG] = 0.0;
    env0[PTR_COMMON_ORIG + 1] = 0.0;
    env0[PTR_COMMON_ORIG + 2] = 0.0;
    let cintx_collapse = collect_cintx_sa10sa01("cg_sa10sa01", &atm, &bas, &env0);
    let vendor_giao = collect_vendor_sa10sa01("int1e_giao_sa10sa01_spinor", &atm, &bas, &env);
    assert_all_components_nonzero(&cintx_collapse, 9, "cg→giao sa01 collapse cintx");
    assert_eq!(
        count_mismatches(&vendor_giao, &cintx_collapse, ATOL, RTOL),
        0,
        "cg_sa10sa01 at common_orig=[0,0,0] MUST collapse to int1e_giao_sa10sa01 \
         (proves the rank-9 gauge term is live, not silently zeroed)"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Sub-wave 1b (Plan 30-01b): the rank-3 Rys+gauge NUCLEAR sa10nucsp gates.
//
// int1e_cg_sa10nucsp / int1e_giao_sa10nucsp — component_rank 3, imaginary
// c2s_si_1ei. The NEW Rys+gauge nuclear engine: the gauge x1i-with-origin fold is
// applied INSIDE the Rys root loop against the nuclear-attraction G2E_* tensor.
// Drives the unified dispatcher on the NON-SQUARE p×d combined gauge∧kappa
// fixture; asserts byte-identity vs vendor at atol=1e-12 + all-3-non-zero +
// the cg→giao collapse witness.
// ═════════════════════════════════════════════════════════════════════════════

/// Build (origin_coords, origin_charges) for the nuclear-attraction engine:
/// sum over atoms, charge −Z at PTR_COORD (point nucleus, libcint int1e_type 2).
/// Cloned from rel_1e_sigma_parity::nuclear_origins (spnucsp arm).
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn nucsp_nuclear_origins(atm: &[i32], env: &[f64]) -> (Vec<f64>, Vec<f64>) {
    use cintx_compat::raw::CHARGE_OF;
    let natm = atm.len() / ATM_SLOTS;
    let mut coords = Vec::with_capacity(natm * 3);
    let mut charges = Vec::with_capacity(natm);
    for a in 0..natm {
        let z = atm[a * ATM_SLOTS + CHARGE_OF];
        let ptr = atm[a * ATM_SLOTS + PTR_COORD] as usize;
        coords.push(env[ptr]);
        coords.push(env[ptr + 1]);
        coords.push(env[ptr + 2]);
        charges.push(-(z as f64));
    }
    (coords, charges)
}

/// Drive the rank-3 nucsp dispatcher for the shell pair (0,1). `op` is
/// `cg_sa10nucsp` or `giao_sa10nucsp`. Returns the flat interleaved-complex rank-3
/// spinor block (3*ni_sp*nj_sp*2). common_orig is read from env (the dispatcher
/// forms dri=ri−common_orig for cg; giao uses origin=[0,0,0] internally).
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_sa10nucsp(op: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_runtime::{BackendIntent, BackendKind};

    let si = extract_shell(0, atm, bas, env);
    let sj = extract_shell(1, atm, bas, env);

    let di = spinor_len(si.l as i32, si.kappa as i32);
    let dj = spinor_len(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;
    // NON-SQUARE block (Pitfall 6 / T-30-01b-05): p(LT,nctr=2)×d(GT).
    assert_ne!(
        ni_sp, nj_sp,
        "nucsp gate requires a NON-SQUARE spinor block"
    );

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend must initialise");

    let common_orig = [
        env[PTR_COMMON_ORIG],
        env[PTR_COMMON_ORIG + 1],
        env[PTR_COMMON_ORIG + 2],
    ];
    // nucsp uses atom-summed nuclear origins (origin_coords/charges), not the single
    // rinv center; pass env[PTR_RINV_ORIG] for signature consistency (ignored by the
    // nucsp arms).
    let rinv_orig = [
        env[PTR_RINV_ORIG],
        env[PTR_RINV_ORIG + 1],
        env[PTR_RINV_ORIG + 2],
    ];
    let (origin_coords, origin_charges) = nucsp_nuclear_origins(atm, env);

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * 2 * 3];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend,
        op,
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
        common_orig,
        rinv_orig,
        &si.exps,
        &sj.exps,
        &si.coeff_row_major,
        &sj.coeff_row_major,
        &origin_coords,
        &origin_charges,
        &mut staging,
    )
    .unwrap_or_else(|e| panic!("{op} rank-3 nucsp spinor launch must succeed: {e:?}"));
    staging
}

/// Vendor rank-3 nucsp collector (REAL FFI shim). out sized 3*ni_sp*nj_sp*2.
#[cfg(has_vendor_libcint)]
fn collect_vendor_sa10nucsp(family: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int1e_cg_sa10nucsp_spinor,
        vendor_int1e_giao_sa10nucsp_spinor,
    };

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2 * family_component_rank(family)];
    match family {
        "int1e_cg_sa10nucsp_spinor" => {
            vendor_int1e_cg_sa10nucsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_giao_sa10nucsp_spinor" => {
            vendor_int1e_giao_sa10nucsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        other => panic!("Sub-wave 1b only drives cg/giao_sa10nucsp; got {other}"),
    };
    out
}

/// cg_sa10nucsp byte-identity vs vendor at atol=1e-12 on the non-square block.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_cg_sa10nucsp() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_sa10nucsp("int1e_cg_sa10nucsp_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_sa10nucsp("cg_sa10nucsp", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "cg_sa10nucsp length (rank 3)");
    assert_all_components_nonzero(&cintx, 3, "cg_sa10nucsp cintx");
    assert_all_components_nonzero(&vendor, 3, "cg_sa10nucsp vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_cg_sa10nucsp: mismatches vs vendored libcint at atol={ATOL} (rank 3)"
    );
}

/// giao_sa10nucsp byte-identity vs vendor at atol=1e-12 on the non-square block.
/// Same gout body as cg; only the gauge builder origin differs ([0,0,0]).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_giao_sa10nucsp() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_sa10nucsp("int1e_giao_sa10nucsp_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_sa10nucsp("giao_sa10nucsp", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "giao_sa10nucsp length (rank 3)");
    assert_all_components_nonzero(&cintx, 3, "giao_sa10nucsp cintx");
    assert_all_components_nonzero(&vendor, 3, "giao_sa10nucsp vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_giao_sa10nucsp: mismatches vs vendored libcint at atol={ATOL} (rank 3)"
    );
}

/// cg→giao differential witness: at common_orig=[0,0,0] with the bra (i) shell at
/// the coordinate origin, dri = ri − 0 = ri = the giao natural center, so
/// cg_sa10nucsp MUST collapse to vendor int1e_giao_sa10nucsp — proving the gauge
/// term is live, not silently zeroed.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_sa10nucsp_cg_giao_collapse() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let si = extract_shell(0, &atm, &bas, &env);
    assert_eq!(
        si.coord,
        [0.0, 0.0, 0.0],
        "collapse witness requires bra at origin"
    );

    let mut env0 = env.clone();
    env0[PTR_COMMON_ORIG] = 0.0;
    env0[PTR_COMMON_ORIG + 1] = 0.0;
    env0[PTR_COMMON_ORIG + 2] = 0.0;
    let cintx_collapse = collect_cintx_sa10nucsp("cg_sa10nucsp", &atm, &bas, &env0);
    let vendor_giao = collect_vendor_sa10nucsp("int1e_giao_sa10nucsp_spinor", &atm, &bas, &env);
    assert_all_components_nonzero(&cintx_collapse, 3, "cg→giao nucsp collapse cintx");
    assert_eq!(
        count_mismatches(&vendor_giao, &cintx_collapse, ATOL, RTOL),
        0,
        "cg_sa10nucsp at common_orig=[0,0,0] MUST collapse to int1e_giao_sa10nucsp \
         (proves the gauge term is live, not silently zeroed)"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Sub-wave 1d (Plan 30-01d): the spg-Rys/London byte-identity gates.
//
// int1e_spgnucsp — rank 3, imaginary c2s_si_1ei, NUCLEAR Rys (int1e_type 2).
// int1e_spgsa01  — rank 9, REAL c2s_si_1e, RINV Rys (int1e_type 1, single center).
// Both COMBINE the 30-01a spgsp London structure (G2E_R0I origin=ri + rirj=ri−rj
// post-multiply; NEITHER reads PTR_COMMON_ORIG) with a Rys root loop. Drives the
// unified dispatcher on the NON-SQUARE p×d combined gauge∧kappa fixture; asserts
// byte-identity vs vendor at atol=1e-12 + all-components-non-zero (Pitfall 3).
// ═════════════════════════════════════════════════════════════════════════════

/// Drive the spg-Rys/London dispatcher for the shell pair (0,1). `op` is `spgnucsp`
/// (rank 3, nuclear, atom-summed origins) or `spgsa01` (rank 9, rinv single center).
/// Neither reads common_orig (the London x1i origin is `ri`, threaded internally).
/// Returns the flat interleaved-complex spinor block (rank*ni_sp*nj_sp*2).
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_spg(op: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_1e::launch_int1e_giao_sigma_family_spinor_pair;
    use cintx_runtime::{BackendIntent, BackendKind};

    let si = extract_shell(0, atm, bas, env);
    let sj = extract_shell(1, atm, bas, env);

    let di = spinor_len(si.l as i32, si.kappa as i32);
    let dj = spinor_len(sj.l as i32, sj.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;
    // NON-SQUARE block (Pitfall 6 / T-30-01d-01): p(LT,nctr=2)×d(GT).
    assert_ne!(ni_sp, nj_sp, "spg gate requires a NON-SQUARE spinor block");

    let rank = if op == "spgsa01" { 9 } else { 3 };

    let backend = ResolvedBackend::from_intent(&BackendIntent {
        backend: BackendKind::Cpu,
        selector: "auto".to_owned(),
    })
    .expect("CPU backend must initialise");

    // spg families do NOT read PTR_COMMON_ORIG (London origin is ri). common_orig
    // is threaded only for signature consistency (ignored by the spg arms).
    let common_orig = [
        env[PTR_COMMON_ORIG],
        env[PTR_COMMON_ORIG + 1],
        env[PTR_COMMON_ORIG + 2],
    ];
    // spgsa01 evaluates at the SINGLE rinv center env[PTR_RINV_ORIG] (int1e_type 1,
    // charge +1) — threaded, not assumed [0,0,0] (the 30-01c lesson). spgnucsp uses
    // the atom-summed nuclear origins instead (int1e_type 2).
    let rinv_orig = [
        env[PTR_RINV_ORIG],
        env[PTR_RINV_ORIG + 1],
        env[PTR_RINV_ORIG + 2],
    ];
    let (origin_coords, origin_charges) = if op == "spgnucsp" {
        nucsp_nuclear_origins(atm, env)
    } else {
        (Vec::new(), Vec::new())
    };

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * 2 * rank];
    launch_int1e_giao_sigma_family_spinor_pair::<f64>(
        &backend,
        op,
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
        common_orig,
        rinv_orig,
        &si.exps,
        &sj.exps,
        &si.coeff_row_major,
        &sj.coeff_row_major,
        &origin_coords,
        &origin_charges,
        &mut staging,
    )
    .unwrap_or_else(|e| panic!("{op} spg-Rys/London spinor launch must succeed: {e:?}"));
    staging
}

/// Vendor spg collector (REAL FFI shim). out sized rank*ni_sp*nj_sp*2.
#[cfg(has_vendor_libcint)]
fn collect_vendor_spg(family: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int1e_spgnucsp_spinor, vendor_int1e_spgsa01_spinor,
    };

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2 * family_component_rank(family)];
    match family {
        "int1e_spgnucsp_spinor" => {
            vendor_int1e_spgnucsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_spgsa01_spinor" => {
            vendor_int1e_spgsa01_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        other => panic!("Sub-wave 1d only drives spgnucsp/spgsa01; got {other}"),
    };
    out
}

/// spgnucsp byte-identity vs vendor at atol=1e-12 on the non-square block (rank 3).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_spgnucsp() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_spg("int1e_spgnucsp_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_spg("spgnucsp", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "spgnucsp length (rank 3)");
    assert_all_components_nonzero(&cintx, 3, "spgnucsp cintx");
    assert_all_components_nonzero(&vendor, 3, "spgnucsp vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_spgnucsp: mismatches vs vendored libcint at atol={ATOL} (rank 3)"
    );
}

/// spgsa01 byte-identity vs vendor at atol=1e-12 on the non-square block. ALL 9
/// rank components must be non-zero (Pitfall 3 / T-30-01d-02: a rank-3 truncation
/// would drop 6 of 9).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn giao_sigma_1e_spgsa01() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();
    let vendor = collect_vendor_spg("int1e_spgsa01_spinor", &atm, &bas, &env);
    let cintx = collect_cintx_spg("spgsa01", &atm, &bas, &env);
    assert_eq!(cintx.len(), vendor.len(), "spgsa01 length (rank 9)");
    assert_all_components_nonzero(&cintx, 9, "spgsa01 cintx");
    assert_all_components_nonzero(&vendor, 9, "spgsa01 vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_spgsa01: mismatches vs vendored libcint at atol={ATOL} (rank 9)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// No-silent-skip integrity for the fully-covered 9-family wave. This test is gated
// under `has_vendor_libcint` so it is absent rather than silently passing without
// CINTX_ORACLE_BUILD_VENDOR=1: the manifest invariants are meaningful only after
// the vendor arms ran.
// ─────────────────────────────────────────────────────────────────────────────
/// Read (oracle_covered, component_rank) for a symbol from the committed manifest
/// lock JSON (the source of truth; the `MANIFEST_ENTRIES`/api_manifest mirror is
/// regenerated from it). Lightweight string scan — avoids a serde dependency and
/// the `vendor-libcint` feature-gate on `manifest_entries()`.
#[cfg(has_vendor_libcint)]
fn manifest_lock_entry(symbol: &str) -> (bool, Option<u32>) {
    let lock = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cintx-ops/generated/compiled_manifest.lock.json"
    ));
    // Per entry, `"component_rank"` precedes the nested `id` object (which holds the
    // `"symbol"` marker) and `"oracle_covered"` follows it. A naive `rfind('{')` lands
    // on the `id` sub-object and misses BOTH fields (the pre-existing 30-01c parser
    // bug that left this test RED). Anchor the entry window on `"arity":` — the FIRST
    // field serialized in every entry — from this entry's arity to the next entry's
    // arity (or file end). That window holds exactly this entry's component_rank and
    // its single oracle_covered.
    let needle = format!("\"symbol\": \"{symbol}\"");
    let sym_pos = lock
        .find(&needle)
        .unwrap_or_else(|| panic!("{symbol} not in lock"));
    let lo = lock[..sym_pos].rfind("\"arity\":").unwrap_or(0);
    let after = sym_pos + needle.len();
    let hi = lock[after..]
        .find("\"arity\":")
        .map(|p| after + p)
        .unwrap_or(lock.len());
    let block = &lock[lo..hi];
    let covered = block.contains("\"oracle_covered\": true");
    let rank = if let Some(p) = block.find("\"component_rank\": \"") {
        let rest = &block[p + "\"component_rank\": \"".len()..];
        rest.split('"').next().and_then(|s| s.parse::<u32>().ok())
    } else {
        None
    };
    (covered, rank)
}

/// No-silent-skip integrity (FULL 9-family Wave-1 gate, Sub-wave 1d). Under the
/// double gate every one of the NINE 1e GIAO×σ families MUST RUN (vendor + cintx
/// arms executed, not skipped), produce non-zero output, AND be byte-identical to
/// vendor at atol=1e-12; AND every row MUST read oracle_covered=true in the manifest
/// lock (sp/nucsp/spg* at rank 3; the three *_sa01 at rank 9). This closes the 1e
/// half of GIAO-03. Gated under `has_vendor_libcint` so it FAILS (compiles out →
/// absent) rather than silently passing without CINTX_ORACLE_BUILD_VENDOR=1
/// (T-30-01d-06): the manifest invariants are meaningless unless the vendor arms ran.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_gauge_kappa_spinor_fixture();

    // (A) Sub-wave 1b: both rank-3 nucsp families RUN under the double gate (no
    //     silent skip), produce all-3 non-zero components, AND are byte-identical.
    for (op, sym) in [
        ("cg_sa10nucsp", "int1e_cg_sa10nucsp_spinor"),
        ("giao_sa10nucsp", "int1e_giao_sa10nucsp_spinor"),
    ] {
        let vendor = collect_vendor_sa10nucsp(sym, &atm, &bas, &env);
        let cintx = collect_cintx_sa10nucsp(op, &atm, &bas, &env);
        assert_eq!(cintx.len(), vendor.len(), "{sym} rank-3 length");
        assert_all_components_nonzero(&cintx, 3, sym);
        assert_all_components_nonzero(&vendor, 3, sym);
        assert_eq!(
            count_mismatches(&vendor, &cintx, ATOL, RTOL),
            0,
            "{sym}: must be byte-identical to vendor at atol={ATOL} (rank 3)"
        );
    }

    // (B) Sub-wave 1c (30-01c root cause fixed): both rank-9 sa01 families RUN (no
    //     silent skip), produce all-9 non-zero components (Pitfall 3 — no rank-3
    //     truncation), AND are byte-identical to vendor (the rinv-center fix).
    for (op, sym) in [
        ("cg_sa10sa01", "int1e_cg_sa10sa01_spinor"),
        ("giao_sa10sa01", "int1e_giao_sa10sa01_spinor"),
    ] {
        let vendor = collect_vendor_sa10sa01(sym, &atm, &bas, &env);
        let cintx = collect_cintx_sa10sa01(op, &atm, &bas, &env);
        assert_eq!(cintx.len(), vendor.len(), "{sym} rank-9 length");
        assert_all_components_nonzero(&cintx, 9, sym);
        assert_all_components_nonzero(&vendor, 9, sym);
        assert_eq!(
            count_mismatches(&vendor, &cintx, ATOL, RTOL),
            0,
            "{sym}: must be byte-identical to vendor at atol={ATOL} (rank 9, rinv-center fix)"
        );
    }

    // (C) Sub-wave 1d spgnucsp (rank 3, NUCLEAR Rys + 8-G London) and spgsa01
    //     (rank 9, RINV Rys + 8-G London): RUNs (no silent skip), produce all
    //     non-zero components, AND byte-identical to vendor.
    {
        let vendor = collect_vendor_spg("int1e_spgnucsp_spinor", &atm, &bas, &env);
        let cintx = collect_cintx_spg("spgnucsp", &atm, &bas, &env);
        assert_eq!(
            cintx.len(),
            vendor.len(),
            "int1e_spgnucsp_spinor rank-3 length"
        );
        assert_all_components_nonzero(&cintx, 3, "int1e_spgnucsp_spinor");
        assert_all_components_nonzero(&vendor, 3, "int1e_spgnucsp_spinor");
        assert_eq!(
            count_mismatches(&vendor, &cintx, ATOL, RTOL),
            0,
            "int1e_spgnucsp_spinor: must be byte-identical to vendor at atol={ATOL} (rank 3)"
        );
    }
    {
        let vendor = collect_vendor_spg("int1e_spgsa01_spinor", &atm, &bas, &env);
        let cintx = collect_cintx_spg("spgsa01", &atm, &bas, &env);
        assert_eq!(
            cintx.len(),
            vendor.len(),
            "int1e_spgsa01_spinor rank-9 length"
        );
        assert_all_components_nonzero(&cintx, 9, "int1e_spgsa01_spinor cintx");
        assert_all_components_nonzero(&vendor, 9, "int1e_spgsa01_spinor vendor");
        assert_eq!(
            count_mismatches(&vendor, &cintx, ATOL, RTOL),
            0,
            "int1e_spgsa01_spinor: must be byte-identical to vendor at atol={ATOL} (rank 9)"
        );
    }

    // (D) Manifest: ALL NINE of the rows are oracle_covered=true. The six
    //     rank-3 sp/nucsp/spgnucsp/spgsp + the three *_sa01 rows at rank 9.
    let covered_rank3 = [
        "int1e_spgsp_spinor",
        "int1e_spgnucsp_spinor",
        "int1e_cg_sa10sp_spinor",
        "int1e_cg_sa10nucsp_spinor",
        "int1e_giao_sa10sp_spinor",
        "int1e_giao_sa10nucsp_spinor",
    ];
    let covered_rank9 = [
        "int1e_cg_sa10sa01_spinor",
        "int1e_giao_sa10sa01_spinor",
        "int1e_spgsa01_spinor",
    ];
    for sym in covered_rank3 {
        let (covered, rank) = manifest_lock_entry(sym);
        assert_eq!(rank, Some(3), "{sym} must carry component_rank 3");
        assert!(
            covered,
            "{sym} must be oracle_covered=true (Wave-1 byte-identity green)"
        );
    }
    for sym in covered_rank9 {
        let (covered, rank) = manifest_lock_entry(sym);
        assert_eq!(
            rank,
            Some(9),
            "{sym} must carry component_rank 9 (Pitfall 3 truncation guard)"
        );
        assert!(
            covered,
            "{sym} must be oracle_covered=true (rank-9 byte-identity green)"
        );
    }
}
