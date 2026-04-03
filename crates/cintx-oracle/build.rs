use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.join("../..");
    let libcint_root = workspace_root.join("libcint-master");
    let cint_h_in = libcint_root.join("include/cint.h.in");
    let cint_funcs_h = libcint_root.join("include/cint_funcs.h");

    println!("cargo:rerun-if-changed=build.rs");
    // Re-run if the vendor gate env var changes (set or unset).
    println!("cargo:rerun-if-env-changed=CINTX_ORACLE_BUILD_VENDOR");
    // Declare the custom cfg flag so rustc doesn't warn about unexpected_cfgs
    println!("cargo::rustc-check-cfg=cfg(has_vendor_libcint)");
    println!("cargo:rerun-if-changed={}", cint_h_in.display());
    println!("cargo:rerun-if-changed={}", cint_funcs_h.display());

    // Emit rerun triggers for the key 1e source files so the build
    // is re-triggered if the vendored source changes.
    for src in &[
        "src/cint_bas.c",
        "src/cint1e.c",
        "src/cint1e_a.c",
        "src/g1e.c",
        "src/cart2sph.c",
        "src/optimizer.c",
        "src/misc.c",
        "src/fmt.c",
        "src/find_roots.c",
        "src/rys_roots.c",
        "src/polyfits.c",
        "src/rys_wheeler.c",
        "src/eigh.c",
        "src/fblas.c",
        "src/c2f.c",
        "src/autocode/intor1.c",
    ] {
        println!("cargo:rerun-if-changed={}", libcint_root.join(src).display());
    }

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

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Process cint.h.in template: replace cmake variables to produce a
    // concrete cint.h without cmake template markers.
    // This gives bindgen and cc a fully resolved header.
    let cint_h_src = fs::read_to_string(&cint_h_in)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cint_h_in.display()));

    // Replace cmake template variables:
    //   @cint_VERSION@      -> "6.1.3"
    //   @cint_SOVERSION@    -> 6   (no quotes — used as integer in the header)
    //   #cmakedefine I8     -> (empty — we want FINT=int, not int64_t)
    //   #cmakedefine CACHE_SIZE_I8 -> (empty — we want CACHE_SIZE_T=FINT=int)
    let cint_h_processed = cint_h_src
        .replace("@cint_VERSION@", "6.1.3")
        .replace("@cint_SOVERSION@", "6")
        .replace("#cmakedefine I8", "/* #cmakedefine I8 disabled — use FINT=int */")
        .replace(
            "#cmakedefine CACHE_SIZE_I8",
            "/* #cmakedefine CACHE_SIZE_I8 disabled — use CACHE_SIZE_T=FINT */",
        );

    let processed_cint_h = out_dir.join("cint.h");
    fs::write(&processed_cint_h, &cint_h_processed)
        .unwrap_or_else(|e| panic!("failed to write processed cint.h: {e}"));

    // Process cint_config.h.in — this cmake template lives in libcint/src and
    // is included by all libcint .c files via `#include "cint_config.h"`.
    // We generate it into OUT_DIR so the `src` include path resolves it.
    let cint_config_h_in = libcint_root.join("src/cint_config.h.in");
    let cint_config_src = fs::read_to_string(&cint_config_h_in)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cint_config_h_in.display()));

    // Replace cmake feature-detect variables:
    //   #cmakedefine HAVE_EXPL        -> /* disabled */ (use stdlib expf)
    //   #cmakedefine HAVE_SQRTL       -> /* disabled */
    //   #cmakedefine HAVE_FABSL       -> /* disabled */
    //   #cmakedefine HAVE_QUADMATH_H  -> /* disabled */
    //   #cmakedefine WITH_RANGE_COULOMB -> /* disabled */
    // We keep the rest of the file (constants, #defines) intact.
    // Note: We include our processed cint.h by redirecting its path.
    let cint_config_processed = cint_config_src
        .replace("#include \"cint.h\"", &format!("#include \"{}\"", processed_cint_h.display()))
        .replace("#cmakedefine HAVE_EXPL", "/* #cmakedefine HAVE_EXPL */")
        .replace("#cmakedefine HAVE_SQRTL", "/* #cmakedefine HAVE_SQRTL */")
        .replace("#cmakedefine HAVE_FABSL", "/* #cmakedefine HAVE_FABSL */")
        .replace("#cmakedefine HAVE_QUADMATH_H", "/* #cmakedefine HAVE_QUADMATH_H */")
        .replace("#cmakedefine WITH_RANGE_COULOMB", "/* #cmakedefine WITH_RANGE_COULOMB */");

    // Write to src/ search path location (OUT_DIR, which is on the include path)
    let processed_cint_config_h = out_dir.join("cint_config.h");
    fs::write(&processed_cint_config_h, &cint_config_processed)
        .unwrap_or_else(|e| panic!("failed to write processed cint_config.h: {e}"));

    // Compile vendored libcint 6.1.3 from source.
    // All 1e integral source files are required for correct evaluation.
    let mut build = cc::Build::new();
    build
        .warnings(false)
        // Suppress type-system errors from K&R-style empty-param function pointer
        // declarations in cint_bas.c. Compiling as gnu89 matches the libcint
        // cmake defaults and accepts `int (*f)()` as "unspecified arguments".
        .flag("-std=gnu89")
        .flag("-Wno-implicit-function-declaration")
        .include(&out_dir) // processed cint.h and cint_config.h live here
        .include(libcint_root.join("include"))
        .include(libcint_root.join("src"))
        .file(libcint_root.join("src/cint_bas.c"))
        .file(libcint_root.join("src/cint1e.c"))
        .file(libcint_root.join("src/cint1e_a.c"))
        .file(libcint_root.join("src/g1e.c"))
        .file(libcint_root.join("src/cart2sph.c"))
        .file(libcint_root.join("src/optimizer.c"))
        .file(libcint_root.join("src/misc.c"))
        .file(libcint_root.join("src/fmt.c"))
        .file(libcint_root.join("src/find_roots.c"))
        .file(libcint_root.join("src/rys_roots.c"))
        .file(libcint_root.join("src/polyfits.c"))
        .file(libcint_root.join("src/rys_wheeler.c"))
        .file(libcint_root.join("src/eigh.c"))
        .file(libcint_root.join("src/fblas.c"))
        .file(libcint_root.join("src/c2f.c"))
        // autocode/intor1.c contains the auto-generated implementations of
        // int1e_kin_{cart,sph,spinor} and many other 1e integrals that are NOT
        // in cint1e.c itself. This file is required for int1e_kin_sph.
        .file(libcint_root.join("src/autocode/intor1.c"))
        .compile("cintx_oracle_vendor");

    println!("cargo:rustc-link-lib=static=cintx_oracle_vendor");
    // libcint uses math.h functions (exp, sqrt, erf, etc.)
    println!("cargo:rustc-link-lib=m");
    // Signal to Rust code that the vendored libcint is available
    println!("cargo:rustc-cfg=has_vendor_libcint");

    // Generate FFI bindings via bindgen.
    // Use the processed cint.h from OUT_DIR so includes resolve correctly.
    let bindings = bindgen::Builder::default()
        .header(cint_funcs_h.to_string_lossy())
        .clang_arg(format!("-I{}", out_dir.display()))
        .clang_arg(format!("-I{}", libcint_root.join("src").display()))
        .allowlist_function("int1e_ovlp_sph|int1e_kin_sph|int1e_nuc_sph|CINTcgto_spheric|CINTinit_optimizer|CINTdel_optimizer")
        .allowlist_type("CINTOpt")
        .generate()
        .expect("failed to generate oracle libcint bindings");

    bindings
        .write_to_file(out_dir.join("oracle_bindings.rs"))
        .expect("failed to write oracle libcint bindings");
}
