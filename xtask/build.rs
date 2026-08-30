fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_vendor_libcint)");
    println!("cargo:rerun-if-env-changed=CINTX_ORACLE_BUILD_VENDOR");
    if std::env::var_os("CINTX_ORACLE_BUILD_VENDOR").is_some() {
        // Mirrors cintx-oracle/build.rs so the reporting command only names the
        // vendor FFI when the dependency compiled and linked it.
        println!("cargo:rustc-cfg=has_vendor_libcint");
    }
}
