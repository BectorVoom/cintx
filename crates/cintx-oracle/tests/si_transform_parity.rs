//! Phase 28 — FND-05 (Gap B2) end-to-end byte-identity parity: the spin-included
//! `cart_to_spinor_si_2d` transform + σ·p `#[cube]` assembler, driven through the
//! `int1e_sp` Spinor kernel path, vs vendored libcint `int1e_sp_spinor` (`c2s_si_1e`).
//!
//! This is THE FND-05 proof. It composes Plan 28-01 (the si_2d transform), Plan 28-02
//! (the σ·p G-tensor assembler), and Plan 28-03 (the manifest row + vendor FFI) into
//! the validated end-to-end path and compares the flat interleaved-complex spinor
//! buffer BYTE-IDENTICALLY to vendored libcint at atol=1e-12 — WITHOUT flipping any
//! manifest coverage flag (D-01: int1e_sp_spinor stays oracle_covered=false; the proof
//! is this dedicated transform test, not a coverage flip).
//!
//! PRIMARY GATE — `build_kappa_spinor_fixture` (D-05 primary):
//!   - NON-SQUARE p × d block (surfaces the KET→BRA orientation transpose; a square
//!     block is transpose-symmetric and HIDES the bug).
//!   - GENERAL CONTRACTION (nctr=2 p shell) — the si_2d path MUST handle nctr>1.
//!   - GENUINE kappa≠0: p kappa=+1 → LT (di = spinor_len(1,+1) = 2), d kappa=−1 → GT
//!     (dj = spinor_len(2,−1) = 6). The FIRST cintx fixture on the non-(4l+2) sizing
//!     path; staging buffer = di*dj*2 = 24 f64 per (ci,cj) sub-block (Spike Target D,
//!     guards Pitfall 3).
//!
//! REALISM CROSS-CHECK — `build_heavy_atom_spinor_fixture` (D-05 secondary, NON-primary
//! gate): a single Hg (Z=80) center p×d spinor pair; asserted finite, not the primary
//! byte-identity gate.
//!
//! Double gate (memory `reference_oracle_vendor_parity_invocation`): the vendor arm is
//! gated on `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1` (the
//! `has_vendor_libcint` cfg). Without both, the vendor bodies compile out. The Phase-27
//! D-10 NO-SILENT-SKIP assertion (`test_no_silent_skip`) FAILS (not skips) if the vendor
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
    /// σ·p assembler reads). The env block is COLUMN-major `[ic*nprim + ip]`
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
// cintx collector: drive the int1e_sp Spinor path (σ·p assembler →
// cart_to_spinor_si_2d) for the single shell pair (0, 1). Returns the flat
// interleaved-complex spinor block, column-major bra-fastest:
//   out[(j*ni_sp + i)*2 + {0:re, 1:im}],  ni_sp = nctr_i*di, nj_sp = nctr_j*dj.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_sp_spinor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::ResolvedBackend;
    use cintx_cubecl::kernels::sigma_p::launch_int1e_sp_spinor_pair;
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

    let mut staging = vec![0.0_f64; ni_sp * nj_sp * 2];
    launch_int1e_sp_spinor_pair::<f64>(
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
        &si.exps,
        &sj.exps,
        &si.coeff_row_major,
        &sj.coeff_row_major,
        &mut staging,
    )
    .expect("int1e_sp spinor launch must succeed");
    staging
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collector (only available with has_vendor_libcint). Single shell pair
// (0, 1); out sized via vendor_CINTcgto_spinor (ni_sp*nj_sp*2 — the kappa≠0
// non-(4l+2) sizing, T-28-04-03 full vendor-expected size).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
fn collect_vendor_sp_spinor(atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{vendor_CINTcgto_spinor, vendor_int1e_sp_spinor};

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let shls: [i32; 2] = [0, 1];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * 2];
    vendor_int1e_sp_spinor(&mut out, &shls, atm, natm, bas, nbas, env);
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
// Sizing guard (Spike Target D / Pitfall 3): the kappa fixture MUST ride the
// non-(4l+2) GT/LT sizing path. Run always under cpu so a sizing regression is
// caught even on a determinism-only (non-vendor) build.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_sizing_non_4l_plus_2() {
    // spinor_len GT/LT sizing on the exact fixture momenta/kappa.
    assert_eq!(spinor_len(1, 1), 2, "p kappa=+1 (LT) → di = 2*1 = 2");
    assert_eq!(spinor_len(2, -1), 6, "d kappa=−1 (GT) → dj = 2*2+2 = 6");
    // Contrast: the kappa=0 sizing the fixture deliberately does NOT use.
    assert_eq!(spinor_len(1, 0), 6, "p kappa=0 would be 4l+2 = 6 (NOT used)");
    assert_eq!(spinor_len(2, 0), 10, "d kappa=0 would be 4l+2 = 10 (NOT used)");

    // The per-(ci,cj) staging sub-block is di*dj*2 = 2*6*2 = 24 f64.
    assert_eq!(spinor_len(1, 1) * spinor_len(2, -1) * 2, 24, "staging sub-block = 24");

    // The fixture really carries p kappa=+1 (nctr=2) and d kappa=−1.
    let (_atm, bas, _env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();
    assert_eq!(bas[ANG_OF], 1, "shell 0 = p");
    assert_eq!(bas[KAPPA_OF], 1, "shell 0 kappa = +1 (LT)");
    assert_eq!(bas[NCTR_OF], 2, "shell 0 nctr = 2 (general contraction)");
    assert_eq!(bas[BAS_SLOTS + ANG_OF], 2, "shell 1 = d");
    assert_eq!(bas[BAS_SLOTS + KAPPA_OF], -1, "shell 1 kappa = −1 (GT)");
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx-only smoke (always runs under cpu, even without the vendor build).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_fixture_evaluates() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();
    let cintx = collect_cintx_sp_spinor(&atm, &bas, &env);
    // ni_sp = nctr_i*di = 2*2 = 4 ; nj_sp = nctr_j*dj = 1*6 = 6 → 4*6*2 = 48.
    assert_eq!(cintx.len(), 4 * 6 * 2, "kappa block = (2*2)*(1*6)*2 = 48");
    assert_any_nonzero(&cintx, "int1e_sp kappa fixture cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_heavy_atom_fixture_finite() {
    // D-05 SECONDARY realism cross-check (non-primary gate): assert finite output.
    let (atm, bas, env) = cintx_oracle::fixtures::build_heavy_atom_spinor_fixture();
    let cintx = collect_cintx_sp_spinor(&atm, &bas, &env);
    // ni_sp = 1*2 = 2, nj_sp = 1*6 = 6 → 2*6*2 = 24.
    assert_eq!(cintx.len(), 2 * 6 * 2, "heavy-atom block = 2*6*2 = 24");
    assert!(
        cintx.iter().all(|v| v.is_finite()),
        "heavy-atom realism cross-check: all spinor output must be finite"
    );
    assert_any_nonzero(&cintx, "int1e_sp heavy-atom fixture cintx");
}

// ─────────────────────────────────────────────────────────────────────────────
// PRIMARY GATE — vendor byte-identity at atol=1e-12 on the kappa fixture.
// Requires has_vendor_libcint + cpu (the double gate).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_sp_kappa_spinor_byte_identity() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();

    let vendor = collect_vendor_sp_spinor(&atm, &bas, &env);
    let cintx = collect_cintx_sp_spinor(&atm, &bas, &env);

    // T-28-04-01: assert the kappa≠0 non-(4l+2) buffer size on both sides.
    assert_eq!(vendor.len(), 4 * 6 * 2, "vendor kappa block = (2*2)*(1*6)*2 = 48");
    assert_eq!(cintx.len(), vendor.len(), "cintx/vendor length must match");

    assert_any_nonzero(&cintx, "int1e_sp kappa cintx");
    assert_any_nonzero(&vendor, "int1e_sp kappa vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_sp_spinor (kappa≠0 non-square nctr>1): {mismatches} mismatches vs vendored \
         libcint c2s_si_1e at atol={ATOL}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Heavy-atom realism cross-check vs vendor (secondary; still byte-identity when
// the vendor arm is live, but NOT the primary gate per D-05).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_sp_heavy_atom_spinor_parity() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_heavy_atom_spinor_fixture();

    let vendor = collect_vendor_sp_spinor(&atm, &bas, &env);
    let cintx = collect_cintx_sp_spinor(&atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_sp heavy-atom cintx");
    assert_any_nonzero(&vendor, "int1e_sp heavy-atom vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_sp_spinor heavy-atom realism cross-check vs vendored libcint at atol={ATOL}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NO-SILENT-SKIP (Phase-27 D-10 / T-28-04-02): when the double-gate env is
// present (has_vendor_libcint), the vendor arm MUST actually execute and produce
// nonzero output — fail (not skip) if it compiled out or returned all-zero. This
// guards verification integrity: the byte-identity gate above cannot silently
// pass on a determinism-only build.
//
// The D-01 manifest-coverage claim is also asserted: int1e_sp_spinor MUST stay
// oracle_covered=false (proof via THIS transform test, not a coverage flip).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();

    // The vendor arm MUST run and produce nonzero output for the kappa fixture.
    let vendor = collect_vendor_sp_spinor(&atm, &bas, &env);
    assert_any_nonzero(
        &vendor,
        "no-silent-skip: vendor int1e_sp_spinor produced all-zero output \
         (fixture skipped / vendor compiled out)",
    );

    // The cintx side of the SAME path must also run (no UnsupportedApi).
    let cintx = collect_cintx_sp_spinor(&atm, &bas, &env);
    assert_any_nonzero(
        &cintx,
        "no-silent-skip: cintx int1e_sp path produced all-zero output (fixture skipped)",
    );

    // D-01: int1e_sp_spinor stays oracle_covered=false — FND-05 is proven by THIS
    // transform test, never by flipping a manifest coverage flag this phase.
    use cintx_ops::generated::MANIFEST_ENTRIES;
    let covered = MANIFEST_ENTRIES
        .iter()
        .find(|e| e.symbol_name == "int1e_sp_spinor")
        .map(|e| e.oracle_covered);
    match covered {
        Some(false) => {}
        Some(true) => panic!(
            "D-01 violated: int1e_sp_spinor reads oracle_covered=true in MANIFEST_ENTRIES — \
             Phase 28 must NOT flip any σ family (all flips → Phase 29)"
        ),
        None => panic!("int1e_sp_spinor is MISSING from MANIFEST_ENTRIES (Plan 28-03 row absent?)"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke: keep the binary non-empty under `--features cpu` alone (the
// vendor parity bodies compile out without CINTX_ORACLE_BUILD_VENDOR=1).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(feature = "cpu", not(has_vendor_libcint)))]
#[test]
fn test_fixtures_build_without_vendor() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_fixture();
    assert_eq!(bas.len() % BAS_SLOTS, 0, "kappa bas rows well-formed");
    assert!(!atm.is_empty() && !env.is_empty(), "kappa fixture populated");

    let (h_atm, h_bas, h_env) = cintx_oracle::fixtures::build_heavy_atom_spinor_fixture();
    assert_eq!(h_bas.len() % BAS_SLOTS, 0, "heavy-atom bas rows well-formed");
    assert!(!h_atm.is_empty() && !h_env.is_empty(), "heavy-atom fixture populated");
}
