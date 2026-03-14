#[link(name = "cint_phase2_cpu", kind = "static")]
#[link(name = "m")]
unsafe extern "C" {
    fn int1e_ovlp_cart();
    fn int1e_ovlp_sph();
    fn int1e_ovlp_spinor();
    fn int2e_cart();
    fn int2e_sph();
    fn int2e_spinor();
    fn int2c2e_ip1_cart();
    fn int2c2e_ip1_sph();
    fn int2c2e_ip1_spinor();
    fn int3c1e_p2_cart();
    fn int3c1e_p2_sph();
    fn int3c1e_p2_spinor();
    fn int3c2e_ip1_cart();
    fn int3c2e_ip1_sph();
    fn int3c2e_ip1_spinor();
}

#[test]
fn cpu_backend_symbols_link() {
    let symbols: &[(&str, *const ())] = &[
        ("int1e_ovlp_cart", int1e_ovlp_cart as *const ()),
        ("int1e_ovlp_sph", int1e_ovlp_sph as *const ()),
        ("int1e_ovlp_spinor", int1e_ovlp_spinor as *const ()),
        ("int2e_cart", int2e_cart as *const ()),
        ("int2e_sph", int2e_sph as *const ()),
        ("int2e_spinor", int2e_spinor as *const ()),
        ("int2c2e_ip1_cart", int2c2e_ip1_cart as *const ()),
        ("int2c2e_ip1_sph", int2c2e_ip1_sph as *const ()),
        ("int2c2e_ip1_spinor", int2c2e_ip1_spinor as *const ()),
        ("int3c1e_p2_cart", int3c1e_p2_cart as *const ()),
        ("int3c1e_p2_sph", int3c1e_p2_sph as *const ()),
        ("int3c1e_p2_spinor", int3c1e_p2_spinor as *const ()),
        ("int3c2e_ip1_cart", int3c2e_ip1_cart as *const ()),
        ("int3c2e_ip1_sph", int3c2e_ip1_sph as *const ()),
        ("int3c2e_ip1_spinor", int3c2e_ip1_spinor as *const ()),
    ];

    for (name, symbol) in symbols {
        assert!(!symbol.is_null(), "symbol `{name}` should be linked");
    }
}
