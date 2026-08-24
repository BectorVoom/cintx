//! Phase 32-03 — the def2 normalization gate.
//!
//! `cintx-basis` reproduces libcint's two-stage normalization (per-primitive
//! `CINTgto_norm`, then per-contraction self-overlap renorm). If either stage
//! is wrong, every integral computed from the emitted `env` is off by a
//! plausible-looking constant rather than obviously broken — so this gate runs
//! before any integral parity test.
//!
//! The check does not need an external fixture: a correctly normalized
//! contracted AO has **unit self-overlap by construction**, so the diagonal of
//! the overlap matrix computed by *vendored libcint* from our `env` must be
//! exactly 1. That makes the vendor, not a recorded file, the authority.
//!
//! It also pins `cintx-basis`'s locally-declared ABI slot constants against
//! `cintx_compat::raw`, which is the reason duplicating them is safe.

#![cfg(has_vendor_libcint)]

use cintx_basis::{AtomSpec, Molecule, StandardBasis, raw as basis_raw, to_raw_arrays};
use cintx_oracle::vendor_ffi;

fn fixtures() -> Vec<(&'static str, Molecule)> {
    let water = |basis| {
        Molecule::new(
            vec![
                AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
                AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
                AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
            ],
            basis,
        )
    };
    let methane = |basis| {
        let d = 0.629;
        Molecule::new(
            vec![
                AtomSpec::from_angstrom("C", [0.0, 0.0, 0.0]).unwrap(),
                AtomSpec::from_angstrom("H", [d, d, d]).unwrap(),
                AtomSpec::from_angstrom("H", [-d, -d, d]).unwrap(),
                AtomSpec::from_angstrom("H", [-d, d, -d]).unwrap(),
                AtomSpec::from_angstrom("H", [d, -d, -d]).unwrap(),
            ],
            basis,
        )
    };
    // Ferrocene-like Fe centre: def2-TZVP iron carries g functions (l = 4),
    // the highest angular momentum in the def2 family.
    let iron = |basis| {
        Molecule::new(
            vec![AtomSpec::from_bohr("Fe", [0.0, 0.0, 0.0]).unwrap()],
            basis,
        )
    };

    let mut out = Vec::new();
    for basis in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        let tag: &'static str = if basis == StandardBasis::Def2Svp {
            "def2-SVP"
        } else {
            "def2-TZVP"
        };
        out.push((tag, water(basis)));
        out.push((tag, methane(basis)));
        out.push((tag, iron(basis)));
    }
    out
}

/// The cintx-basis ABI slot constants must equal the compat crate's, which is
/// what makes duplicating them in a leaf data crate safe.
#[test]
fn raw_slot_constants_match_compat() {
    use cintx_compat::raw;
    assert_eq!(basis_raw::CHARGE_OF, raw::CHARGE_OF);
    assert_eq!(basis_raw::PTR_COORD, raw::PTR_COORD);
    assert_eq!(basis_raw::NUC_MOD_OF, raw::NUC_MOD_OF);
    assert_eq!(basis_raw::PTR_ZETA, raw::PTR_ZETA);
    assert_eq!(basis_raw::ATM_SLOTS, raw::ATM_SLOTS);
    assert_eq!(basis_raw::ATOM_OF, raw::ATOM_OF);
    assert_eq!(basis_raw::ANG_OF, raw::ANG_OF);
    assert_eq!(basis_raw::NPRIM_OF, raw::NPRIM_OF);
    assert_eq!(basis_raw::NCTR_OF, raw::NCTR_OF);
    assert_eq!(basis_raw::KAPPA_OF, raw::KAPPA_OF);
    assert_eq!(basis_raw::PTR_EXP, raw::PTR_EXP);
    assert_eq!(basis_raw::PTR_COEFF, raw::PTR_COEFF);
    assert_eq!(basis_raw::BAS_SLOTS, raw::BAS_SLOTS);
    assert_eq!(basis_raw::PTR_ENV_START, raw::PTR_ENV_START);
    assert_eq!(basis_raw::POINT_NUC, raw::POINT_NUC);
}

/// THE normalization gate: vendored libcint, fed our `env`, must report a
/// unit diagonal for the AO overlap matrix on every fixture.
#[test]
fn def2_contracted_aos_have_unit_self_overlap_under_vendor_libcint() {
    for (tag, molecule) in fixtures() {
        let arrays = to_raw_arrays(&molecule).expect("raw arrays should build");
        let natm = arrays.natm() as i32;
        let nbas = arrays.nbas() as i32;

        for shell in 0..arrays.nbas() {
            let ao_count = vendor_ffi::vendor_cgto_spheric(shell as i32, &arrays.bas) as usize;
            assert!(ao_count > 0, "{tag}: shell {shell} reported zero AOs");

            let mut block = vec![0.0_f64; ao_count * ao_count];
            let status = vendor_ffi::vendor_int1e_ovlp_sph(
                &mut block,
                &[shell as i32, shell as i32],
                &arrays.atm,
                natm,
                &arrays.bas,
                nbas,
                &arrays.env,
            );
            assert_ne!(status, 0, "{tag}: vendor overlap returned all-zero block");

            for ao in 0..ao_count {
                let diagonal = block[ao * ao_count + ao];
                assert!(
                    (diagonal - 1.0).abs() < 1e-12,
                    "{tag}: shell {shell} AO {ao} self-overlap {diagonal} != 1 — \
                     normalization does not match libcint"
                );
            }
        }
    }
}

/// Every def2 element in the catalog must produce a `BasisSet` and raw arrays
/// without error, and no shell may exceed the angular momentum the kernels
/// declare support for.
#[test]
fn every_catalog_element_builds() {
    for basis in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        for &z in basis.table().keys() {
            let molecule = Molecule::new(
                vec![
                    AtomSpec::from_bohr(cintx_basis::element::symbol(z).unwrap(), [0.0, 0.0, 0.0])
                        .unwrap(),
                ],
                basis,
            );
            let arrays = to_raw_arrays(&molecule)
                .unwrap_or_else(|e| panic!("{} Z={z} raw build failed: {e}", basis.name()));
            assert!(
                arrays.nbas() > 0,
                "{} Z={z} produced no shells",
                basis.name()
            );
            molecule
                .to_basis_set()
                .unwrap_or_else(|e| panic!("{} Z={z} BasisSet build failed: {e}", basis.name()));

            let max_l = (0..arrays.nbas())
                .map(|s| arrays.bas[s * basis_raw::BAS_SLOTS + basis_raw::ANG_OF])
                .max()
                .unwrap();
            assert!(
                max_l <= 4,
                "{} Z={z} has l={max_l}; def2 is documented to top out at g (l=4)",
                basis.name()
            );
        }
    }
}

/// Direct check of stage 1 against the vendor's own `CINTgto_norm`, so a
/// regression in the Lanczos `ln_gamma` cannot hide behind stage 2's
/// self-overlap rescaling.
#[test]
fn gto_norm_matches_vendor_cintgto_norm() {
    for l in 0..=4_i32 {
        for &alpha in &[
            0.012_f64, 0.1, 0.44453796, 1.0, 1.9622572, 13.010701, 130.70932, 5000.0,
        ] {
            let ours = cintx_basis::gto_norm(l, alpha);
            let vendor = vendor_ffi::vendor_CINTgto_norm(l, alpha);
            let tolerance = 1e-13 * vendor.abs().max(1.0);
            assert!(
                (ours - vendor).abs() <= tolerance,
                "gto_norm({l}, {alpha}): ours={ours} vendor={vendor}"
            );
        }
    }
}

/// Stage 2 must actually fire: the raw catalog coefficients are *not* already
/// normalized, so the emitted `env` must differ from them. A no-op stage 2
/// would still pass the unit-diagonal test only if the input happened to be
/// normalized — this pins that it did not.
#[test]
fn emitted_env_coefficients_differ_from_raw_catalog() {
    let molecule = Molecule::new(
        vec![AtomSpec::from_bohr("O", [0.0, 0.0, 0.0]).unwrap()],
        StandardBasis::Def2Tzvp,
    );
    let arrays = to_raw_arrays(&molecule).unwrap();
    let catalog = &StandardBasis::Def2Tzvp.table()[&8];

    let mut any_changed = false;
    for (shell, block) in catalog.iter().enumerate() {
        let ptr = arrays.bas[shell * basis_raw::BAS_SLOTS + basis_raw::PTR_COEFF] as usize;
        let len = block.nprim() * block.nctr;
        for (index, &raw_value) in block.coefficients.iter().enumerate().take(len) {
            if (arrays.env[ptr + index] - raw_value).abs() > 1e-15 {
                any_changed = true;
            }
        }
    }
    assert!(
        any_changed,
        "normalization did not modify any coefficient — stage 1/2 are not running"
    );
}
