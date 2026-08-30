//! Precision-critical constants must preserve the vendored libcint bit pattern
//! across every numerical implementation, rather than be recomputed from a
//! language-level mathematical constant.

#[test]
fn pie4_is_one_verbatim_libcint_bit_pattern_everywhere() {
    let expected = 0.785_398_163_397_448_279_00_f64.to_bits();
    assert_eq!(cintx_simd::boys::LIBCINT_PIE4.to_bits(), expected);
    assert_eq!(cintx_cubecl::math::rys::LIBCINT_PIE4.to_bits(), expected);
}
