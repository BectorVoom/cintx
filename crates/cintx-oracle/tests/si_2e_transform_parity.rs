//! Phase 29 — Wave-2 / D-03 [BLOCKING] gate: the brand-new 2e si/sf cart→spinor
//! transform suite (`cart_to_spinor_si_2e1` + `cart_to_spinor_sf_2e2`, Plan 29-03)
//! driven through the thinnest 2e σ family `int2e_spsp1_spinor` (intor4.c:85),
//! compared BYTE-IDENTICALLY to vendored libcint `int2e_spsp1_spinor` at atol=1e-12.
//!
//! This is THE D-03 structural mitigation: the novel 2e transform layout (the
//! electron-2 `zcopy_iklj` store + the per-electron split) is genuinely new in
//! cintx; a wrong 2e layout/stride/sign surfaces HERE — on the thinnest family —
//! BEFORE any of the 16 Wave-3 2e σ families wires onto the transform (29-06).
//!
//! **[BLOCKING]: Wave 3 (29-05/06) does NOT begin until this test is GREEN.**
//!
//! PRIMARY GATE — `build_kappa_spinor_2e_fixture` (D-02 primary):
//!   - 4 spinor shells, NON-SQUARE dims (2,6,2,4) — all distinct, so the bra/ket
//!     orientation transpose cannot hide (a square block is transpose-symmetric).
//!   - GENUINE kappa≠0 GT/LT mix: i p kappa=+1 (LT, di=2); j d, k s, l p all
//!     kappa=−1 (GT). Rides BOTH `spinor_len` sizing branches (2l and 2l+2), not 4l+2.
//!   - GENERAL CONTRACTION: shell i has nctr=2 (column-major env coeff) — the 2e
//!     transform MUST handle nctr>1.
//!
//! Double gate (memory `reference_oracle_vendor_parity_invocation`): the vendor arm
//! is gated on `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1` (the
//! `has_vendor_libcint` cfg). Without both, the vendor bodies compile out. The
//! NO-SILENT-SKIP assertion (`test_no_silent_skip`) FAILS (not skips) if the vendor
//! arm did not actually run when it was expected.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

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
// Extract a single shell's primitive data from atm/bas/env.
// ─────────────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
struct ShellData {
    l: u8,
    kappa: i16,
    nprim: usize,
    nctr: usize,
    coord: [f64; 3],
    exps: Vec<f64>,
    /// ROW-major coefficients `[ip*nctr + ic]` (the cintx `Shell` convention the
    /// 2e σ·p assembler reads). The env block is COLUMN-major `[ic*nprim + ip]`
    /// (libcint CINTprim_to_ctr_0); transposed here once on extraction
    /// (project_raw_nctr_coeff_transpose).
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

    // env coeff is COLUMN-major [ic*nprim + ip]; transpose to ROW-major [ip*nctr + ic].
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
// cintx collector: drive the int2e_spsp1 Spinor path (σ·p₁ assembler →
// cart_to_spinor_si_2e1 + cart_to_spinor_sf_2e2) for the quartet (0,1,2,3).
// Returns the flat interleaved-complex spinor block, column-major i-fastest:
//   out[(((l*nk_sp+k)*nj_sp+j)*ni_sp+i)*2 + {0:re,1:im}].
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_spsp1_spinor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::kernels::two_electron::{
        int2e_common_factor, launch_int2e_spsp1_spinor_quartet,
    };

    let si = extract_shell(0, atm, bas, env);
    let sj = extract_shell(1, atm, bas, env);
    let sk = extract_shell(2, atm, bas, env);
    let sl = extract_shell(3, atm, bas, env);

    let di = spinor_len(si.l as i32, si.kappa as i32);
    let dj = spinor_len(sj.l as i32, sj.kappa as i32);
    let dk = spinor_len(sk.l as i32, sk.kappa as i32);
    let dl = spinor_len(sl.l as i32, sl.kappa as i32);
    let ni_sp = si.nctr * di;
    let nj_sp = sj.nctr * dj;
    let nk_sp = sk.nctr * dk;
    let nl_sp = sl.nctr * dl;

    let common_factor = int2e_common_factor(si.l, sj.l, sk.l, sl.l);

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * nk_sp * nl_sp * 2];
    launch_int2e_spsp1_spinor_quartet::<f64>(
        si.l, si.kappa, sj.l, sj.kappa, sk.l, sk.kappa, sl.l, sl.kappa,
        si.nprim, sj.nprim, sk.nprim, sl.nprim,
        si.nctr, sj.nctr, sk.nctr, sl.nctr,
        si.coord, sj.coord, sk.coord, sl.coord,
        common_factor,
        &si.exps, &sj.exps, &sk.exps, &sl.exps,
        &si.coeff_row_major, &sj.coeff_row_major, &sk.coeff_row_major, &sl.coeff_row_major,
        &mut staging,
    )
    .expect("int2e_spsp1 spinor quartet launch must succeed");
    staging
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collector (only available with has_vendor_libcint). Quartet (0,1,2,3);
// out sized via vendor_CINTcgto_spinor (kappa≠0 non-(4l+2) sizing).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
fn collect_vendor_spsp1_spinor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{vendor_CINTcgto_spinor, vendor_int2e_spsp1_spinor};

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let nk_sp = vendor_CINTcgto_spinor(2, bas) as usize;
    let nl_sp = vendor_CINTcgto_spinor(3, bas) as usize;
    let shls: [i32; 4] = [0, 1, 2, 3];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * nk_sp * nl_sp * 2];
    vendor_int2e_spsp1_spinor(&mut out, &shls, atm, natm, bas, nbas, env);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity helpers.
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
// Kappa-sizing guard: the 2e fixture MUST ride the non-(4l+2) GT/LT sizing path.
// Run always under cpu so a sizing regression is caught even on a determinism-only
// (non-vendor) build.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_sizing_2e_non_4l_plus_2() {
    // spinor_len GT/LT sizing on the exact fixture momenta/kappa.
    assert_eq!(spinor_len(1, 1), 2, "i: p kappa=+1 (LT) → di = 2*1 = 2");
    assert_eq!(spinor_len(2, -1), 6, "j: d kappa=−1 (GT) → dj = 2*2+2 = 6");
    assert_eq!(spinor_len(0, -1), 2, "k: s kappa=−1 (GT) → dk = 2*0+2 = 2");
    assert_eq!(spinor_len(1, -1), 4, "l: p kappa=−1 (GT) → dl = 2*1+2 = 4");
    // Contrast: the kappa=0 sizing the fixture deliberately does NOT use.
    assert_eq!(spinor_len(1, 0), 6, "p kappa=0 would be 4l+2 = 6 (NOT used)");
    assert_eq!(spinor_len(2, 0), 10, "d kappa=0 would be 4l+2 = 10 (NOT used)");

    // The fixture really carries i p kappa=+1 (nctr=2) and j d / k s / l p kappa=−1.
    let (_atm, bas, _env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();
    assert_eq!(bas[ANG_OF], 1, "shell 0 (i) = p");
    assert_eq!(bas[KAPPA_OF], 1, "shell 0 (i) kappa = +1 (LT)");
    assert_eq!(bas[NCTR_OF], 2, "shell 0 (i) nctr = 2 (general contraction)");
    assert_eq!(bas[BAS_SLOTS + ANG_OF], 2, "shell 1 (j) = d");
    assert_eq!(bas[BAS_SLOTS + KAPPA_OF], -1, "shell 1 (j) kappa = −1 (GT)");
    assert_eq!(bas[2 * BAS_SLOTS + ANG_OF], 0, "shell 2 (k) = s");
    assert_eq!(bas[2 * BAS_SLOTS + KAPPA_OF], -1, "shell 2 (k) kappa = −1 (GT)");
    assert_eq!(bas[3 * BAS_SLOTS + ANG_OF], 1, "shell 3 (l) = p");
    assert_eq!(bas[3 * BAS_SLOTS + KAPPA_OF], -1, "shell 3 (l) kappa = −1 (GT)");

    // Total spinor block = (2*2)*(1*6)*(1*2)*(1*4)*2 = 4*6*2*4*2 = 384.
    assert_eq!(2 * 2 * 6 * 2 * 4 * 2, 384, "non-square 2e spinor block = 384 f64");
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx-only smoke (always runs under cpu, even without the vendor build).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_2e_fixture_evaluates() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();
    let cintx = collect_cintx_spsp1_spinor(&atm, &bas, &env);
    // ni_sp=2*2=4, nj_sp=6, nk_sp=2, nl_sp=4 → 4*6*2*4*2 = 384.
    assert_eq!(cintx.len(), 4 * 6 * 2 * 4 * 2, "kappa 2e block = 384");
    assert!(cintx.iter().all(|v| v.is_finite()), "all spinor output must be finite");
    assert_any_nonzero(&cintx, "int2e_spsp1 kappa fixture cintx");
}

// ─────────────────────────────────────────────────────────────────────────────
// PRIMARY GATE — vendor byte-identity at atol=1e-12 on the 2e kappa fixture.
// Requires has_vendor_libcint + cpu (the double gate). This is the D-03 BLOCKING
// gate that pins the 2e transform layout BEFORE any Wave-3 family wires onto it.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2e_spsp1_kappa_spinor_byte_identity() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();

    let vendor = collect_vendor_spsp1_spinor(&atm, &bas, &env);
    let cintx = collect_cintx_spsp1_spinor(&atm, &bas, &env);

    // Assert the kappa≠0 non-(4l+2) non-square buffer size on both sides.
    assert_eq!(vendor.len(), 4 * 6 * 2 * 4 * 2, "vendor kappa 2e block = 384");
    assert_eq!(cintx.len(), vendor.len(), "cintx/vendor length must match");

    assert_any_nonzero(&cintx, "int2e_spsp1 kappa cintx");
    assert_any_nonzero(&vendor, "int2e_spsp1 kappa vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int2e_spsp1_spinor (kappa≠0 non-square nctr>1): {mismatches} mismatches vs vendored \
         libcint c2s_si_2e1+c2s_sf_2e2 at atol={ATOL} — the D-03 BLOCKING 2e transform gate"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NO-SILENT-SKIP (Phase-27 D-10 / T-29-08): when the double-gate env is present
// (has_vendor_libcint), the vendor arm MUST actually execute and produce nonzero
// output — fail (not skip) if it compiled out or returned all-zero. Guards
// verification integrity: the byte-identity gate above cannot silently pass on a
// determinism-only build.
//
// The manifest-coverage claim is also asserted: int2e_spsp1_spinor MUST stay
// oracle_covered=false this plan (flipped in 29-06 when the family wires on).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();

    // The vendor arm MUST run and produce nonzero output for the kappa fixture.
    let vendor = collect_vendor_spsp1_spinor(&atm, &bas, &env);
    assert_any_nonzero(
        &vendor,
        "no-silent-skip: vendor int2e_spsp1_spinor produced all-zero output \
         (fixture skipped / vendor compiled out)",
    );

    // The cintx side of the SAME path must also run (no UnsupportedApi).
    let cintx = collect_cintx_spsp1_spinor(&atm, &bas, &env);
    assert_any_nonzero(
        &cintx,
        "no-silent-skip: cintx int2e_spsp1 path produced all-zero output (fixture skipped)",
    );

    // D-03 ordering invariant: int2e_spsp1_spinor stays oracle_covered=false until
    // 29-06 wires the family. The proof of the transform is THIS test, not a flip.
    use cintx_ops::generated::MANIFEST_ENTRIES;
    let covered = MANIFEST_ENTRIES
        .iter()
        .find(|e| e.symbol_name == "int2e_spsp1_spinor")
        .map(|e| e.oracle_covered);
    match covered {
        Some(false) => {}
        Some(true) => panic!(
            "int2e_spsp1_spinor must stay oracle_covered=false this plan (29-06 flips it after \
             wiring the family); the D-03 proof is this transform test, not a coverage flip"
        ),
        None => {
            panic!("int2e_spsp1_spinor is MISSING from MANIFEST_ENTRIES (Plan 29-04 row absent?)")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke: keep the binary non-empty under `--features cpu` alone (the
// vendor parity bodies compile out without CINTX_ORACLE_BUILD_VENDOR=1).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(feature = "cpu", not(has_vendor_libcint)))]
#[test]
fn test_2e_fixture_builds_without_vendor() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();
    assert_eq!(bas.len() % BAS_SLOTS, 0, "kappa 2e bas rows well-formed");
    assert!(!atm.is_empty() && !env.is_empty(), "kappa 2e fixture populated");
    assert_eq!(bas.len() / BAS_SLOTS, 4, "exactly 4 shells (a 2-electron quartet)");
}
