//! Phase 29 Wave 1 — 1e Group-4 relativistic σ spinor parity scaffold (REL-01/02).
//!
//! Drives the seven 1e Group-4 σ families
//!   `int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor`
//! against vendored libcint 6.1.3 at atol=1e-12 on the Phase-28 kappa fixture
//! (p kappa=+1 LT nctr=2 × d kappa=−1 GT — NON-SQUARE, kappa≠0 GT/LT, nctr>1).
//!
//! STATUS — RED SCAFFOLD (Plan 29-01): the cintx launcher arms for these seven
//! families are NOT yet wired (Plan 29-02 adds the `one_electron.rs` Spinor arms +
//! flips `oracle_covered`). This file:
//!   - registers the per-family VENDOR collectors against the real Plan-29-01 FFI
//!     shims (`vendor_int1e_*_spinor`), which DO link under the double gate;
//!   - leaves each `collect_cintx_<fam>` as a RED stub that `unimplemented!`s with a
//!     `TODO(29-02)` pointing at the launcher-arm work;
//!   - marks every per-family byte-identity gate `#[ignore]` ("RED until 29-02")
//!     so the scaffold COMPILES and the suite is green at the Wave-1 boundary while
//!     the gates remain explicitly pending;
//!   - keeps the always-on `test_kappa_sizing_non_4l_plus_2` sizing guard and the
//!     `test_no_silent_skip` verification-integrity assertion live.
//!
//! Per-family transform map (RESEARCH §Per-Family gout & Transform Map), wired in 29-02:
//!   spsp                              → c2s_sf_1e   (scalar; ∇² folds to spin-free)
//!   spnucsp / sprinvsp / srsr / srnucsr → c2s_si_1e  (real bra-σ-mix)
//!   sr / sigma                        → c2s_si_1ei  (imaginary-ket bra-σ-mix — the new si_2di)
//!
//! Double gate (memory `reference_oracle_vendor_parity_invocation`): the vendor arm
//! is gated on `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1` (the
//! `has_vendor_libcint` cfg). Without both, the vendor bodies compile out. The
//! Phase-27 D-10 NO-SILENT-SKIP assertion FAILS (not skips) if the vendor arm did
//! not run when it was expected.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

/// The seven 1e Group-4 σ family symbols this scaffold gates (REL-01/02).
#[allow(dead_code)]
const REL_1E_FAMILIES: &[&str] = &[
    "int1e_spsp_spinor",
    "int1e_spnucsp_spinor",
    "int1e_sprinvsp_spinor",
    "int1e_srsr_spinor",
    "int1e_srnucsr_spinor",
    "int1e_sr_spinor",
    "int1e_sigma_spinor",
];

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
// Parity helpers (cloned from si_transform_parity.rs).
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
// cintx collectors — RED STUBS (Plan 29-01). The int1e_<op> Spinor launcher arms
// land in Plan 29-02 (one_electron.rs); until then these `unimplemented!`. Keep the
// signatures + the per-family transform note so 29-02 only fills in the body.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_rel_1e(_family: &str, _atm: &[i32], _bas: &[i32], _env: &[f64]) -> Vec<f64> {
    // TODO(29-02): wire the int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor
    // Spinor launcher arms in one_electron.rs (σ·p assembler → cart_to_spinor_si_2d /
    // cart_to_spinor_sf_2d / cart_to_spinor_si_2di per the transform map), then replace
    // this stub with the real per-family launch + scatter and flip oracle_covered.
    unimplemented!(
        "RED scaffold (Plan 29-01): cintx 1e Group-4 σ launcher arm for {_family} \
         is wired in Plan 29-02"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collectors — REAL (Plan 29-01 FFI shims). Single shell pair (0, 1);
// out sized via vendor_CINTcgto_spinor (ni_sp*nj_sp*2 — component_rank=1 for all
// Group-4, kappa≠0 non-(4l+2) sizing).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
fn collect_vendor_rel_1e(family: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int1e_sigma_spinor, vendor_int1e_spnucsp_spinor,
        vendor_int1e_sprinvsp_spinor, vendor_int1e_spsp_spinor, vendor_int1e_sr_spinor,
        vendor_int1e_srnucsr_spinor, vendor_int1e_srsr_spinor,
    };

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2];
    match family {
        "int1e_spsp_spinor" => {
            vendor_int1e_spsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_spnucsp_spinor" => {
            vendor_int1e_spnucsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_sprinvsp_spinor" => {
            vendor_int1e_sprinvsp_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_srsr_spinor" => {
            vendor_int1e_srsr_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_srnucsr_spinor" => {
            vendor_int1e_srnucsr_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        "int1e_sr_spinor" => vendor_int1e_sr_spinor(&mut out, &shls, atm, natm, bas, nbas, env),
        "int1e_sigma_spinor" => {
            vendor_int1e_sigma_spinor(&mut out, &shls, atm, natm, bas, nbas, env)
        }
        other => panic!("unknown REL-1e family: {other}"),
    };
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Sizing guard (Pitfall 3): the kappa fixture MUST ride the non-(4l+2) GT/LT
// sizing path. Runs always under cpu so a sizing regression is caught even on a
// determinism-only (non-vendor) build.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_sizing_non_4l_plus_2() {
    assert_eq!(spinor_len(1, 1), 2, "p kappa=+1 (LT) → di = 2*1 = 2");
    assert_eq!(spinor_len(2, -1), 6, "d kappa=−1 (GT) → dj = 2*2+2 = 6");
    assert_eq!(spinor_len(1, 0), 6, "p kappa=0 would be 4l+2 = 6 (NOT used)");
    assert_eq!(spinor_len(2, 0), 10, "d kappa=0 would be 4l+2 = 10 (NOT used)");
    assert_eq!(spinor_len(1, 1) * spinor_len(2, -1) * 2, 24, "staging sub-block = 24");

    let (_atm, bas, _env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();
    assert_eq!(bas[ANG_OF], 1, "shell 0 = p");
    assert_eq!(bas[KAPPA_OF], 1, "shell 0 kappa = +1 (LT)");
    assert_eq!(bas[NCTR_OF], 2, "shell 0 nctr = 2 (general contraction)");
    assert_eq!(bas[BAS_SLOTS + ANG_OF], 2, "shell 1 = d");
    assert_eq!(bas[BAS_SLOTS + KAPPA_OF], -1, "shell 1 kappa = −1 (GT)");
}

// ─────────────────────────────────────────────────────────────────────────────
// PRIMARY GATES — per-family vendor byte-identity at atol=1e-12 on the kappa
// fixture. `#[ignore]` until Plan 29-02 wires the cintx launcher arms (the
// collect_cintx_rel_1e stub `unimplemented!`s today). Run with `--ignored`
// after 29-02 to drive the byte-identity comparison.
// ─────────────────────────────────────────────────────────────────────────────
macro_rules! rel_1e_byte_identity_gate {
    ($test_name:ident, $family:literal) => {
        #[cfg(has_vendor_libcint)]
        #[cfg(feature = "cpu")]
        #[test]
        #[ignore = "RED until Plan 29-02 wires the cintx int1e σ launcher arms"]
        fn $test_name() {
            let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();

            let vendor = collect_vendor_rel_1e($family, &atm, &bas, &env);
            let cintx = collect_cintx_rel_1e($family, &atm, &bas, &env);

            assert_eq!(cintx.len(), vendor.len(), "cintx/vendor length must match for {}", $family);
            assert_any_nonzero(&cintx, concat!($family, " kappa cintx"));
            assert_any_nonzero(&vendor, concat!($family, " kappa vendor"));

            let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
            assert_eq!(
                mismatches, 0,
                "{} (kappa≠0 non-square nctr>1): {} mismatches vs vendored libcint at atol={}",
                $family, mismatches, ATOL
            );
        }
    };
}

rel_1e_byte_identity_gate!(test_spsp_kappa_byte_identity, "int1e_spsp_spinor");
rel_1e_byte_identity_gate!(test_spnucsp_kappa_byte_identity, "int1e_spnucsp_spinor");
rel_1e_byte_identity_gate!(test_sprinvsp_kappa_byte_identity, "int1e_sprinvsp_spinor");
rel_1e_byte_identity_gate!(test_srsr_kappa_byte_identity, "int1e_srsr_spinor");
rel_1e_byte_identity_gate!(test_srnucsr_kappa_byte_identity, "int1e_srnucsr_spinor");
rel_1e_byte_identity_gate!(test_sr_kappa_byte_identity, "int1e_sr_spinor");
rel_1e_byte_identity_gate!(test_sigma_kappa_byte_identity, "int1e_sigma_spinor");

// ─────────────────────────────────────────────────────────────────────────────
// NO-SILENT-SKIP (Phase-27 D-10): when the double-gate env is present
// (has_vendor_libcint), the vendor arm MUST actually execute and produce nonzero
// output for every family — fail (not skip) if it compiled out or returned
// all-zero. This guards verification integrity even while the cintx side is RED.
//
// The Wave-1 manifest-coverage contract is also asserted: every REL-1e family is
// registered in MANIFEST_ENTRIES and STAYS oracle_covered=false this plan (29-02
// flips each only after its byte-identity gate is green).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    use cintx_ops::generated::MANIFEST_ENTRIES;

    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();

    for &family in REL_1E_FAMILIES {
        // The vendor arm MUST run and produce nonzero output for the kappa fixture.
        let vendor = collect_vendor_rel_1e(family, &atm, &bas, &env);
        assert_any_nonzero(
            &vendor,
            &format!(
                "no-silent-skip: vendor {family} produced all-zero output \
                 (fixture skipped / vendor compiled out)"
            ),
        );

        // Wave-1 coverage contract: the row exists and stays oracle_covered=false
        // (29-02 flips it after the byte-identity gate is green).
        let covered = MANIFEST_ENTRIES
            .iter()
            .find(|e| e.symbol_name == family)
            .map(|e| e.oracle_covered);
        match covered {
            Some(false) => {}
            Some(true) => panic!(
                "Wave-1 contract violated: {family} reads oracle_covered=true in MANIFEST_ENTRIES \
                 — 29-01 registers spinor-only rows oracle_covered=false; the flip is 29-02"
            ),
            None => panic!("{family} is MISSING from MANIFEST_ENTRIES (Plan 29-01 row absent?)"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke: keep the binary non-empty under `--features cpu` alone (the
// vendor parity bodies compile out without CINTX_ORACLE_BUILD_VENDOR=1). Also
// asserts the 29-01 manifest rows are present + spinor-only + oracle_covered=false.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(feature = "cpu", not(has_vendor_libcint)))]
#[test]
fn test_rel_1e_rows_registered_without_vendor() {
    use cintx_ops::generated::MANIFEST_ENTRIES;

    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();
    assert_eq!(bas.len() % BAS_SLOTS, 0, "kappa bas rows well-formed");
    assert!(!atm.is_empty() && !env.is_empty(), "kappa fixture populated");

    for &family in REL_1E_FAMILIES {
        let entry = MANIFEST_ENTRIES
            .iter()
            .find(|e| e.symbol_name == family)
            .unwrap_or_else(|| panic!("{family} missing from MANIFEST_ENTRIES (Plan 29-01 row)"));
        assert_eq!(entry.component_rank, "1", "{family} must be component_rank=1");
        assert_eq!(entry.forms, &["spinor"], "{family} must be spinor-only");
        assert!(!entry.oracle_covered, "{family} must stay oracle_covered=false in 29-01");
    }
}
