//! Phase 29 — Wave-3 REL-03/04 parity scaffold: byte-identity of the remaining 2e
//! Group-4 relativistic σ spinor families against vendored libcint 6.1.3, on the
//! `build_kappa_spinor_2e_fixture` (D-02) at atol=1e-12.
//!
//! Families covered (all component_rank=1, transform pair from 29-RESEARCH §2e map):
//!   REL-03 (intor4.c):  srsr1 (si_2e1+sf_2e2), spsp1spsp2 (si_2e1+si_2e2),
//!                       srsr1srsr2 (si_2e1+si_2e2)
//!   REL-04 (gaunt1.c):  ssp1ssp2, ssp1sps2, sps1ssp2, sps1sps2 (si_2e1i+si_2e2i)
//!   REL-04 (dkb.c):     spv1, vsp1 (si_2e1+sf_2e2); spv1spv2, vsp1spv2,
//!                       spv1vsp2, vsp1vsp2, spv1spsp2, vsp1spsp2 (si_2e1+si_2e2)
//!
//! STATE: LIVE (Plan 29-06). All 15 family launcher arms are wired in
//! `launch_two_electron_typed` (REL-03 Task 1 + REL-04 Task 2); the cintx collector
//! drives each via `rel2e_family_dispatch`, and every per-family byte-identity gate
//! runs under the double gate at atol=1e-12.
//!
//! ALWAYS-ON:
//!   - `test_kappa_sizing_2e_non_4l_plus_2` — GT/LT sizing on the fixture momenta.
//!   - `test_all_rel_2e_rows_registered` — all 15 manifest rows present, spinor-only,
//!     component_rank=1. (oracle_covered flips true per family in Task 3.)
//!   - `test_no_silent_skip` — under the double gate every vendor arm MUST execute
//!     and produce nonzero output (FAIL, not skip).
//!
//! Double gate (memory `reference_oracle_vendor_parity_invocation`): vendor arms are
//! gated on `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1` (`has_vendor_libcint`).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

/// The 15 remaining 2e Group-4 spinor families registered in 29-05 (the per-family
/// byte-identity wiring lands in 29-06). The single source of truth for both the
/// manifest-registration assertion and the no-silent-skip vendor sweep.
#[allow(dead_code)]
const REL_2E_FAMILIES: &[&str] = &[
    // REL-03 (intor4.c)
    "int2e_srsr1_spinor",
    "int2e_spsp1spsp2_spinor",
    "int2e_srsr1srsr2_spinor",
    // REL-04 (gaunt1.c)
    "int2e_ssp1ssp2_spinor",
    "int2e_ssp1sps2_spinor",
    "int2e_sps1ssp2_spinor",
    "int2e_sps1sps2_spinor",
    // REL-04 (dkb.c)
    "int2e_spv1_spinor",
    "int2e_vsp1_spinor",
    "int2e_spv1spv2_spinor",
    "int2e_vsp1spv2_spinor",
    "int2e_spv1vsp2_spinor",
    "int2e_vsp1vsp2_spinor",
    "int2e_spv1spsp2_spinor",
    "int2e_vsp1spsp2_spinor",
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
// Parity helpers (cloned from si_2e_transform_parity.rs).
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
// Vendor collector: drives an arbitrary 2e Group-4 spinor family on the quartet
// (0,1,2,3). The shim is selected by symbol name. `out` sized via
// vendor_CINTcgto_spinor (kappa≠0 non-(4l+2) sizing). Available NOW — the shims
// + gaunt1.c/dkb.c build wiring all land in 29-05.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
fn collect_vendor_family(symbol: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_oracle::vendor_ffi::{
        vendor_CINTcgto_spinor, vendor_int2e_sps1sps2_spinor, vendor_int2e_sps1ssp2_spinor,
        vendor_int2e_spsp1spsp2_spinor, vendor_int2e_spv1spsp2_spinor, vendor_int2e_spv1spv2_spinor,
        vendor_int2e_spv1vsp2_spinor, vendor_int2e_spv1_spinor, vendor_int2e_srsr1srsr2_spinor,
        vendor_int2e_srsr1_spinor, vendor_int2e_ssp1sps2_spinor, vendor_int2e_ssp1ssp2_spinor,
        vendor_int2e_vsp1spsp2_spinor, vendor_int2e_vsp1spv2_spinor, vendor_int2e_vsp1vsp2_spinor,
        vendor_int2e_vsp1_spinor,
    };
    use cintx_compat::raw::ATM_SLOTS;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ni_sp = vendor_CINTcgto_spinor(0, bas) as usize;
    let nj_sp = vendor_CINTcgto_spinor(1, bas) as usize;
    let nk_sp = vendor_CINTcgto_spinor(2, bas) as usize;
    let nl_sp = vendor_CINTcgto_spinor(3, bas) as usize;
    let shls: [i32; 4] = [0, 1, 2, 3];

    let mut out = vec![0.0_f64; ni_sp * nj_sp * nk_sp * nl_sp * 2];
    let f: fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32 = match symbol {
        "int2e_srsr1_spinor" => vendor_int2e_srsr1_spinor,
        "int2e_spsp1spsp2_spinor" => vendor_int2e_spsp1spsp2_spinor,
        "int2e_srsr1srsr2_spinor" => vendor_int2e_srsr1srsr2_spinor,
        "int2e_ssp1ssp2_spinor" => vendor_int2e_ssp1ssp2_spinor,
        "int2e_ssp1sps2_spinor" => vendor_int2e_ssp1sps2_spinor,
        "int2e_sps1ssp2_spinor" => vendor_int2e_sps1ssp2_spinor,
        "int2e_sps1sps2_spinor" => vendor_int2e_sps1sps2_spinor,
        "int2e_spv1_spinor" => vendor_int2e_spv1_spinor,
        "int2e_vsp1_spinor" => vendor_int2e_vsp1_spinor,
        "int2e_spv1spv2_spinor" => vendor_int2e_spv1spv2_spinor,
        "int2e_vsp1spv2_spinor" => vendor_int2e_vsp1spv2_spinor,
        "int2e_spv1vsp2_spinor" => vendor_int2e_spv1vsp2_spinor,
        "int2e_vsp1vsp2_spinor" => vendor_int2e_vsp1vsp2_spinor,
        "int2e_spv1spsp2_spinor" => vendor_int2e_spv1spsp2_spinor,
        "int2e_vsp1spsp2_spinor" => vendor_int2e_vsp1spsp2_spinor,
        other => panic!("unknown REL-2e vendor family: {other}"),
    };
    f(&mut out, &shls, atm, natm, bas, nbas, env);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collector: RED STUB until 29-06.
//
// TODO(29-06): wire each family's launcher arm in
// `cintx_cubecl::kernels::two_electron::launch_two_electron_typed` (per the
// 29-RESEARCH §2e transform map), then replace this stub with the real launch
// (mirroring `collect_cintx_spsp1_spinor` in si_2e_transform_parity.rs):
//   srsr1/spv1/vsp1          → si_2e1 + sf_2e2
//   spsp1spsp2/srsr1srsr2    → si_2e1 + si_2e2
//   *spv2/*vsp2/*spsp2       → si_2e1 + si_2e2
//   ssp1ssp2/ssp1sps2/       → si_2e1i + si_2e2i (both imaginary)
//   sps1ssp2/sps1sps2
// drop `#[ignore]` from each per-family gate and flip oracle_covered=true once green.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[allow(dead_code)]
struct ShellData {
    l: u8,
    kappa: i16,
    nprim: usize,
    nctr: usize,
    coord: [f64; 3],
    exps: Vec<f64>,
    /// ROW-major coefficients `[ip*nctr + ic]` (the cintx `Shell` convention). The env
    /// block is COLUMN-major `[ic*nprim + ip]`; transposed here on extraction
    /// (project_raw_nctr_coeff_transpose).
    coeff_row_major: Vec<f64>,
}

#[cfg(feature = "cpu")]
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

/// Drive a REL-03/04 2e σ family on the quartet (0,1,2,3) through the cintx launcher.
/// The operator name (`int2e_<op>_spinor` → `<op>`) selects the (gout, e1×e2) pairing
/// via `rel2e_family_dispatch`. Returns the flat interleaved-complex spinor block.
#[cfg(feature = "cpu")]
#[allow(dead_code)]
fn collect_cintx_family(symbol: &str, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    use cintx_cubecl::kernels::two_electron::{
        int2e_common_factor, launch_rel2e_sigma_spinor_quartet, rel2e_family_dispatch,
    };

    // "int2e_srsr1_spinor" → "srsr1".
    let op = symbol
        .strip_prefix("int2e_")
        .and_then(|s| s.strip_suffix("_spinor"))
        .unwrap_or_else(|| panic!("unexpected REL-2e symbol shape: {symbol}"));
    let (gout_kind, e1, e2) = rel2e_family_dispatch(op)
        .unwrap_or_else(|| panic!("no rel2e dispatch for {op} (29-06 wiring missing?)"));

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
    launch_rel2e_sigma_spinor_quartet::<f64>(
        gout_kind, e1, e2,
        si.l, si.kappa, sj.l, sj.kappa, sk.l, sk.kappa, sl.l, sl.kappa,
        si.nprim, sj.nprim, sk.nprim, sl.nprim,
        si.nctr, sj.nctr, sk.nctr, sl.nctr,
        si.coord, sj.coord, sk.coord, sl.coord,
        common_factor,
        &si.exps, &sj.exps, &sk.exps, &sl.exps,
        &si.coeff_row_major, &sj.coeff_row_major, &sk.coeff_row_major, &sl.coeff_row_major,
        &mut staging,
    )
    .unwrap_or_else(|e| panic!("{symbol} cintx launch failed: {e:?}"));
    staging
}

// ─────────────────────────────────────────────────────────────────────────────
// ALWAYS-ON: kappa-sizing guard — the 2e fixture rides the non-(4l+2) GT/LT path.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_kappa_sizing_2e_non_4l_plus_2() {
    assert_eq!(spinor_len(1, 1), 2, "i: p kappa=+1 (LT) → di = 2*1 = 2");
    assert_eq!(spinor_len(2, -1), 6, "j: d kappa=−1 (GT) → dj = 2*2+2 = 6");
    assert_eq!(spinor_len(0, -1), 2, "k: s kappa=−1 (GT) → dk = 2*0+2 = 2");
    assert_eq!(spinor_len(1, -1), 4, "l: p kappa=−1 (GT) → dl = 2*1+2 = 4");
    // Contrast: the kappa=0 sizing the fixture deliberately does NOT use.
    assert_eq!(spinor_len(1, 0), 6, "p kappa=0 would be 4l+2 = 6 (NOT used)");
    assert_eq!(spinor_len(2, 0), 10, "d kappa=0 would be 4l+2 = 10 (NOT used)");

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

    // Total spinor block = (2*2)*(1*6)*(1*2)*(1*4)*2 = 384.
    assert_eq!(2 * 2 * 6 * 2 * 4 * 2, 384, "non-square 2e spinor block = 384 f64");
}

// ─────────────────────────────────────────────────────────────────────────────
// ALWAYS-ON: all 15 remaining 2e Group-4 rows are registered spinor-only,
// component_rank=1, oracle_covered=false (the 29-06 pre-flip state). Guards the
// component_rank-truncation landmine and the SC#5 spinor-only invariant.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "cpu")]
#[test]
fn test_all_rel_2e_rows_registered() {
    use cintx_ops::generated::MANIFEST_ENTRIES;
    for &sym in REL_2E_FAMILIES {
        let entry = MANIFEST_ENTRIES
            .iter()
            .find(|e| e.symbol_name == sym)
            .unwrap_or_else(|| panic!("REL-2e family {sym} MISSING from MANIFEST_ENTRIES (29-05 row absent?)"));
        assert_eq!(entry.arity, 4, "{sym}: arity must be 4 (2e quartet)");
        assert_eq!(entry.canonical_family, "2e", "{sym}: canonical_family must be 2e");
        assert_eq!(
            entry.component_rank, "1",
            "{sym}: component_rank must be 1 (σ fold is internal to c2s, NOT an output axis)"
        );
        assert!(entry.complex_output, "{sym}: complex_output must be true (spinor)");
        assert_eq!(
            entry.forms,
            &["spinor"],
            "{sym}: forms must be [spinor] only (no cart/sph over-claim, SC#5)"
        );
        assert!(
            !entry.oracle_covered,
            "{sym}: oracle_covered must stay false this plan (29-06 flips it after parity green)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ALWAYS-ON under the double gate: NO-SILENT-SKIP (Phase-27 D-10 / T-29-10).
// Every vendor arm MUST execute and produce nonzero output for the kappa fixture
// (FAIL, not skip). This is the proof that 29-05 Task 1 actually linked a real
// libcint driver for EACH family — including the gaunt1.c/dkb.c REL-04 families
// that had no vendor symbol before this plan.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();

    for &sym in REL_2E_FAMILIES {
        let vendor = collect_vendor_family(sym, &atm, &bas, &env);
        assert_eq!(
            vendor.len(),
            4 * 6 * 2 * 4 * 2,
            "{sym}: vendor kappa 2e block must be 384 (non-square GT/LT sizing)"
        );
        assert_any_nonzero(
            &vendor,
            &format!(
                "no-silent-skip: vendor {sym} produced all-zero output (driver not linked / \
                 gaunt1.c/dkb.c build wiring missing / fixture skipped)"
            ),
        );

        // 29-06 ordering invariant: stays oracle_covered=false until its launcher
        // is wired and its byte-identity gate flips green.
        use cintx_ops::generated::MANIFEST_ENTRIES;
        let covered = MANIFEST_ENTRIES
            .iter()
            .find(|e| e.symbol_name == sym)
            .map(|e| e.oracle_covered);
        assert_eq!(
            covered,
            Some(false),
            "{sym}: must stay oracle_covered=false this plan (29-06 flips after wiring)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-family byte-identity gates — all LIVE (29-06): each family's launcher arm is
// wired in `launch_two_electron_typed`, and the cintx collector drives it via
// `rel2e_family_dispatch`. Each gate runs the cintx vs vendor comparison under the
// double gate at atol=1e-12 on the non-square kappa fixture.
// ─────────────────────────────────────────────────────────────────────────────

/// LIVE byte-identity gate (29-06): cintx launcher wired, runs under the double gate.
macro_rules! rel_2e_byte_identity_gate_live {
    ($test_name:ident, $symbol:literal) => {
        #[cfg(has_vendor_libcint)]
        #[cfg(feature = "cpu")]
        #[test]
        fn $test_name() {
            let (atm, bas, env) = cintx_oracle::fixtures::build_kappa_spinor_2e_fixture();
            let vendor = collect_vendor_family($symbol, &atm, &bas, &env);
            let cintx = collect_cintx_family($symbol, &atm, &bas, &env);
            assert_eq!(vendor.len(), cintx.len(), "{}: length mismatch", $symbol);
            assert_any_nonzero(&vendor, concat!($symbol, " vendor"));
            assert_any_nonzero(&cintx, concat!($symbol, " cintx"));
            let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
            assert_eq!(
                mismatches, 0,
                "{}: {} mismatches vs vendored libcint at atol={} (29-06 byte-identity gate)",
                $symbol, mismatches, ATOL
            );
        }
    };
}

// REL-03 (intor4.c) — LIVE (29-06 Task 1).
rel_2e_byte_identity_gate_live!(test_srsr1_byte_identity, "int2e_srsr1_spinor");
rel_2e_byte_identity_gate_live!(test_spsp1spsp2_byte_identity, "int2e_spsp1spsp2_spinor");

rel_2e_byte_identity_gate_live!(test_srsr1srsr2_byte_identity, "int2e_srsr1srsr2_spinor");
// REL-04 (gaunt1.c)
rel_2e_byte_identity_gate_live!(test_ssp1ssp2_byte_identity, "int2e_ssp1ssp2_spinor");
rel_2e_byte_identity_gate_live!(test_ssp1sps2_byte_identity, "int2e_ssp1sps2_spinor");
rel_2e_byte_identity_gate_live!(test_sps1ssp2_byte_identity, "int2e_sps1ssp2_spinor");
rel_2e_byte_identity_gate_live!(test_sps1sps2_byte_identity, "int2e_sps1sps2_spinor");
// REL-04 (dkb.c)
rel_2e_byte_identity_gate_live!(test_spv1_byte_identity, "int2e_spv1_spinor");
rel_2e_byte_identity_gate_live!(test_vsp1_byte_identity, "int2e_vsp1_spinor");
rel_2e_byte_identity_gate_live!(test_spv1spv2_byte_identity, "int2e_spv1spv2_spinor");
rel_2e_byte_identity_gate_live!(test_vsp1spv2_byte_identity, "int2e_vsp1spv2_spinor");
rel_2e_byte_identity_gate_live!(test_spv1vsp2_byte_identity, "int2e_spv1vsp2_spinor");
rel_2e_byte_identity_gate_live!(test_vsp1vsp2_byte_identity, "int2e_vsp1vsp2_spinor");
rel_2e_byte_identity_gate_live!(test_spv1spsp2_byte_identity, "int2e_spv1spsp2_spinor");
rel_2e_byte_identity_gate_live!(test_vsp1spsp2_byte_identity, "int2e_vsp1spsp2_spinor");

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


