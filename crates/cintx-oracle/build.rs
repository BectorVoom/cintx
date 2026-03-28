use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.join("../..");
    let libcint_root = workspace_root.join("libcint-master");
    let cint_h = libcint_root.join("include/cint.h.in");
    let cint_funcs_h = libcint_root.join("include/cint_funcs.h");
    let misc_c = libcint_root.join("src/misc.c");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", cint_h.display());
    println!("cargo:rerun-if-changed={}", cint_funcs_h.display());
    println!("cargo:rerun-if-changed={}", misc_c.display());

    // Keep bindgen and cc wiring in place for oracle-harness parity work.
    // The heavy vendor build stays behind an env gate so default `cargo test`
    // remains fast and deterministic in CI sandboxes.
    let _bindgen_probe = bindgen::Builder::default()
        .header(cint_funcs_h.to_string_lossy())
        .allowlist_function("cint.*");
    let _cc_probe = cc::Build::new();

    if env::var_os("CINTX_ORACLE_BUILD_VENDOR").is_none() {
        return;
    }

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .include(libcint_root.join("include"))
        .include(libcint_root.join("src"))
        .file(misc_c)
        .compile("cintx_oracle_vendor_probe");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let bindings = bindgen::Builder::default()
        .header(cint_funcs_h.to_string_lossy())
        .allowlist_function("cint.*")
        .generate()
        .expect("failed to generate oracle libcint bindings");
    bindings
        .write_to_file(out_dir.join("oracle_bindings.rs"))
        .expect("failed to write oracle libcint bindings");
}
