//! Oracle parity test for the cart-to-spherical bra transform `CINTc2s_bra_sph` (HELP-02).
//!
//! libcint 6.1.3 ground truth (`libcint-master/src/cart2sph.c`):
//!   `CINTc2s_bra_sph` dispatches on l. The oracle build does NOT define PYPZPX.
//!     l=0  s_bra_cart2spheric: `return gcart;` (does NOT write gsph) — identity.
//!     l=1  p_bra_cart2spheric (non-PYPZPX): `return gcart;` — px,py,pz identity.
//!     l>=2 d/f/g_bra_cart2spheric: apply coeffs, write gsph ket-blocked
//!          (per ket: write nsph=2l+1, advance gsph+=nsph, gcart+=ncart), `return gsph`.
//!   => For l<2 the returned pointer aliases the `cart` INPUT, NOT `sph`.
//!
//! The vendor wrapper must copy the RETURNED pointer into `sph` so the harness
//! never reads zero-init for l<2 (Defect A). cintx must apply the real per-l
//! transform, not the identity (Defect B).
//!
//! Vendor parity is double-gated: it only runs under `--features cpu` AND env
//! `CINTX_ORACLE_BUILD_VENDOR=1` (which makes build.rs set `has_vendor_libcint`).
//! Without both, the vendor body is cfg'd out and the file compiles to the
//! non-vendor smoke test only.

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[allow(dead_code)]
const ATOL: f64 = 1e-12;

/// Number of Cartesian components for angular momentum l: (l+1)(l+2)/2.
#[allow(dead_code)]
fn ncart(l: i32) -> usize {
    let l = l.max(0) as usize;
    (l + 1) * (l + 2) / 2
}

/// Number of spherical components for angular momentum l: 2l+1.
#[allow(dead_code)]
fn nsph(l: i32) -> usize {
    2 * l.max(0) as usize + 1
}

/// Expected l=2 d-transform of cart=[1.0..6.0], shared by smoke + reference.
#[allow(dead_code)]
fn d_transform_l2(cart: &[f64]) -> [f64; 5] {
    [
        1.092548430592079070 * cart[1],
        1.092548430592079070 * cart[4],
        -0.315391565252520002 * cart[0] - 0.315391565252520002 * cart[3]
            + 0.630783130505040012 * cart[5],
        1.092548430592079070 * cart[2],
        0.546274215296039535 * cart[0] - 0.546274215296039535 * cart[3],
    ]
}

/// Non-vendor smoke test so the file is not a pure no-op without the vendor build.
#[cfg(feature = "cpu")]
#[test]
fn cintc2s_bra_sph_smoke() {
    let cart = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut sph = vec![0.0f64; 5]; // nsph(2)
    cintx_compat::transform::CINTc2s_bra_sph(&mut sph, 1, &cart, 2).unwrap();

    let expected = d_transform_l2(&cart);
    for m in 0..5 {
        assert!(sph[m].is_finite(), "sph[{m}] must be finite");
        assert!(
            (sph[m] - expected[m]).abs() < ATOL,
            "smoke d-transform sph[{m}] = {} expected {}",
            sph[m],
            expected[m]
        );
    }
}

/// Vendor parity: cintx `CINTc2s_bra_sph` vs libcint 6.1.3 over the **whole**
/// `l = 0..=C2S_LMAX` range, `nket` in {1, 2}.
///
/// The sweep used to stop at `l = 4`, because that is where cintx's
/// hand-transcribed coefficient tables stopped — and above it the accessor
/// returned `0.0`, so an `l >= 5` shell came back silently zeroed with an `Ok`
/// status at any Rys order. The table is now generated from libcint's own
/// `g_trans_cart2sph[]` for `l = 0..=15` (`xtask gen-c2s-table`), so this gate
/// covers every `l` the transform claims to support. `l >= 5` has no
/// hand-checked reference in-tree, which is exactly why it is compared against
/// the vendor's own routine here.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn cintc2s_bra_sph_matches_vendor() {
    use cintx_cubecl::transform::c2s::C2S_LMAX;
    use cintx_oracle::vendor_ffi;

    let mut mismatches = 0usize;
    let mut report = String::new();
    let mut high_l_nonzero = 0usize;

    for l in 0i32..=i32::from(C2S_LMAX) {
        let nc = ncart(l);
        let ns = nsph(l);
        for &nket in &[1i32, 2i32] {
            let nk = nket as usize;
            // deterministic cart of len nket*ncart(l)
            let cart: Vec<f64> = (0..nk * nc).map(|i| (i + 1) as f64 * 0.1).collect();

            let mut cintx_out = vec![0.0f64; nk * ns];
            let mut vendor_out = vec![0.0f64; nk * ns];

            cintx_compat::transform::CINTc2s_bra_sph(&mut cintx_out, nket, &cart, l).unwrap();
            vendor_ffi::vendor_CINTc2s_bra_sph(&mut vendor_out, nket, &cart, l);

            for idx in 0..nk * ns {
                let c = cintx_out[idx];
                let v = vendor_out[idx];
                let diff = (c - v).abs();
                if diff > ATOL {
                    mismatches += 1;
                    report.push_str(&format!(
                        "  (l={l}, nket={nket}, idx={idx}): cintx={c} vendor={v} diff={diff}\n"
                    ));
                }
                if l >= 5 && c != 0.0 {
                    high_l_nonzero += 1;
                }
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "CINTc2s_bra_sph vendor parity mismatches ({mismatches}):\n{report}"
    );
    // Agreement alone would be satisfied by both sides returning zero, which is
    // the failure mode this sweep was extended to catch.
    assert!(
        high_l_nonzero > 0,
        "no non-zero l>=5 output was produced; the transform is still zeroing \
         the range this gate exists to cover"
    );
}

/// Above `C2S_LMAX` the wrapper refuses rather than zeroing.
#[cfg(feature = "cpu")]
#[test]
fn cintc2s_bra_sph_refuses_above_the_table_ceiling() {
    use cintx_cubecl::transform::c2s::C2S_LMAX;

    let l = i32::from(C2S_LMAX) + 1;
    let cart = vec![1.0_f64; ncart(l)];
    let mut sph = vec![0.0_f64; nsph(l)];
    let status = cintx_compat::transform::CINTc2s_bra_sph(&mut sph, 1, &cart, l);
    assert!(
        status.is_err(),
        "l={l} is past the coefficient table and must be refused, not zeroed"
    );
}
