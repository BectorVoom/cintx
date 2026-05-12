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
    // Phase 19 D-01: PySCF nr_ecp vendor cfg — emitted by the parallel
    // cc::Build chain below when CINTX_ORACLE_BUILD_VENDOR=1.
    println!("cargo::rustc-check-cfg=cfg(has_vendor_pyscf_nr_ecp)");
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
        "src/cint2e.c",
        "src/g2e.c",
        "src/cint2c2e.c",
        "src/g2c2e.c",
        "src/cint3c1e.c",
        "src/g3c1e.c",
        "src/cint3c2e.c",
        "src/g3c2e.c",
        "src/autocode/intor2.c",
        "src/autocode/intor3.c",
        "src/autocode/intor4.c",
        "src/autocode/int3c1e.c",
        "src/autocode/int3c2e.c",
        "src/cint4c1e.c",
        "src/g4c1e.c",
        // F12/STG/YP integral source files (required for int2e_stg_sph and related).
        "src/cint2e_f12.c",
        "src/g2e_f12.c",
        "src/stg_roots.c",
        // Derivative gout implementations required by cint2e_f12.c.
        // CINTgout2e_int2e_ip1 is in grad2.c; ipip1/ipvip1/ip1ip2 are in hess.c.
        "src/autocode/grad2.c",
        "src/autocode/hess.c",
        // Phase 14 unstable-source family source files.
        "src/cint1e_grids.c",
        "src/g1e_grids.c",
        "src/autocode/int1e_grids1.c",
        "src/breit.c",
        "src/cint3c1e_a.c",
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
        // 2e+ integral source files (required for int2e_sph and related families).
        .file(libcint_root.join("src/cint2e.c"))
        .file(libcint_root.join("src/g2e.c"))
        .file(libcint_root.join("src/cint2c2e.c"))
        .file(libcint_root.join("src/g2c2e.c"))
        .file(libcint_root.join("src/cint3c1e.c"))
        .file(libcint_root.join("src/g3c1e.c"))
        .file(libcint_root.join("src/cint3c2e.c"))
        .file(libcint_root.join("src/g3c2e.c"))
        .file(libcint_root.join("src/autocode/intor2.c"))
        .file(libcint_root.join("src/autocode/intor3.c"))
        .file(libcint_root.join("src/autocode/intor4.c"))
        .file(libcint_root.join("src/autocode/int3c1e.c"))
        .file(libcint_root.join("src/autocode/int3c2e.c"))
        // 4c1e integral source files (required for int4c1e_sph and int4c1e_cart).
        .file(libcint_root.join("src/cint4c1e.c"))
        .file(libcint_root.join("src/g4c1e.c"))
        // F12/STG/YP integral source files (required for int2e_stg_sph and related families).
        // stg_roots.c includes roots_xw.dat from the same src/ directory.
        .file(libcint_root.join("src/cint2e_f12.c"))
        .file(libcint_root.join("src/g2e_f12.c"))
        .file(libcint_root.join("src/stg_roots.c"))
        // Derivative gout implementations: CINTgout2e_int2e_ip1 (grad2.c),
        // CINTgout2e_int2e_ipip1/ipvip1/ip1ip2 (hess.c). These are declared
        // extern in cint2e_f12.c but defined in the autocode files.
        .file(libcint_root.join("src/autocode/grad2.c"))
        .file(libcint_root.join("src/autocode/hess.c"))
        // Phase 14 unstable-source family source files.
        // cint1e_grids.c + g1e_grids.c: int1e_grids* (NGRIDS/PTR_GRIDS env handling).
        // autocode/int1e_grids1.c: int1e_grids_ip/ipvip/spvsp/ipip derivative wrappers.
        // breit.c: int2e_breit_r1p2_spinor and int2e_breit_r2p2_spinor.
        // cint3c1e_a.c: int3c1e_r2/r4/r6_origk and ip1 derivatives (origk family).
        // cint1e_a.c already present above (origi family symbols defined there).
        // cint3c2e.c already present above (ssc family).
        .file(libcint_root.join("src/cint1e_grids.c"))
        .file(libcint_root.join("src/g1e_grids.c"))
        .file(libcint_root.join("src/autocode/int1e_grids1.c"))
        .file(libcint_root.join("src/breit.c"))
        .file(libcint_root.join("src/cint3c1e_a.c"))
        .compile("cintx_oracle_vendor");

    println!("cargo:rustc-link-lib=static=cintx_oracle_vendor");
    // libcint uses math.h functions (exp, sqrt, erf, etc.)
    println!("cargo:rustc-link-lib=m");
    // Signal to Rust code that the vendored libcint is available
    println!("cargo:rustc-cfg=has_vendor_libcint");

    // Generate a supplemental header for functions declared only in .c files
    // (not in cint_funcs.h). These are the basic sph variants of multi-center integrals,
    // plus the spinor variants for 2c2e, 3c1e, and 3c2e (int2e_spinor is in cint_funcs.h).
    // We use `extern CINTIntegralFunction` declarations matching the pattern in cint_funcs.h.
    let suppl_h_content = format!(
        r#"
#include "{cint_funcs}"
/* Supplemental declarations for variants not in cint_funcs.h */
extern CINTIntegralFunction int2c2e_sph;
extern CINTIntegralFunction int2c2e_cart;
extern CINTIntegralFunction int3c1e_sph;
extern CINTIntegralFunction int3c1e_cart;
extern CINTIntegralFunction int3c1e_p2_sph;
extern CINTIntegralFunction int3c2e_sph;
extern CINTIntegralFunction int3c2e_cart;
extern CINTIntegralFunction int4c1e_sph;
extern CINTIntegralFunction int4c1e_cart;
/* Multi-center spinor variants: int2e_spinor is in cint_funcs.h;
   int2c2e_spinor, int3c1e_spinor, int3c2e_spinor are only in .c files. */
extern CINTIntegralFunction int2c2e_spinor;
extern CINTIntegralFunction int3c1e_spinor;
extern CINTIntegralFunction int3c2e_spinor;
/* F12/STG/YP integral declarations: all in cint2e_f12.c, not in cint_funcs.h */
extern CINTIntegralFunction int2e_stg_sph;
extern CINTIntegralFunction int2e_stg_ip1_sph;
extern CINTIntegralFunction int2e_stg_ipip1_sph;
extern CINTIntegralFunction int2e_stg_ipvip1_sph;
extern CINTIntegralFunction int2e_stg_ip1ip2_sph;
extern CINTIntegralFunction int2e_yp_sph;
extern CINTIntegralFunction int2e_yp_ip1_sph;
extern CINTIntegralFunction int2e_yp_ipip1_sph;
extern CINTIntegralFunction int2e_yp_ipvip1_sph;
extern CINTIntegralFunction int2e_yp_ip1ip2_sph;
/* Phase 14 unstable-source family declarations */
/* origi (1e) — in cint1e_a.c, not in cint_funcs.h */
extern CINTIntegralFunction int1e_r2_origi_sph;
extern CINTIntegralFunction int1e_r4_origi_sph;
extern CINTIntegralFunction int1e_r2_origi_ip2_sph;
extern CINTIntegralFunction int1e_r4_origi_ip2_sph;
/* grids (1e) — int1e_grids_sph NOT in cint_funcs.h (Pitfall 4) */
extern CINTIntegralFunction int1e_grids_sph;
extern CINTIntegralFunction int1e_grids_ip_sph;
extern CINTIntegralFunction int1e_grids_ipvip_sph;
extern CINTIntegralFunction int1e_grids_spvsp_sph;
extern CINTIntegralFunction int1e_grids_ipip_sph;
/* breit (2e spinor) — in breit.c */
extern CINTIntegralFunction int2e_breit_r1p2_spinor;
extern CINTIntegralFunction int2e_breit_r2p2_spinor;
/* origk (3c1e) — in cint3c1e_a.c */
extern CINTIntegralFunction int3c1e_r2_origk_sph;
extern CINTIntegralFunction int3c1e_r4_origk_sph;
extern CINTIntegralFunction int3c1e_r6_origk_sph;
extern CINTIntegralFunction int3c1e_ip1_r2_origk_sph;
extern CINTIntegralFunction int3c1e_ip1_r4_origk_sph;
extern CINTIntegralFunction int3c1e_ip1_r6_origk_sph;
/* ssc (3c2e) — in cint3c2e.c */
extern CINTIntegralFunction int3c2e_sph_ssc;
/* Phase 19 ECP (PySCF nr_ecp) declarations — link with cintx_pyscf_nr_ecp.
   Signatures match the libcint CINTIntegralFunction shape; ecpbas + necpbas
   are read from env[AS_ECPBAS_OFFSET] / env[AS_NECPBAS] inside the function
   body (nr_ecp.c:6205-6206 and ECPscalar_cart:6248-6249). */
extern int ECPscalar_sph(double *out, int *dims, int *shls,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         void *opt, double *cache);
extern int ECPscalar_cart(double *out, int *dims, int *shls,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          void *opt, double *cache);
extern int ECPscalar_ipnuc_sph(double *out, int *dims, int *shls,
                               int *atm, int natm, int *bas, int nbas, double *env,
                               void *opt, double *cache);
extern int ECPscalar_ipnuc_cart(double *out, int *dims, int *shls,
                                int *atm, int natm, int *bas, int nbas, double *env,
                                void *opt, double *cache);
"#,
        cint_funcs = cint_funcs_h.display()
    );
    let suppl_h = out_dir.join("cintx_oracle_suppl.h");
    fs::write(&suppl_h, &suppl_h_content)
        .unwrap_or_else(|e| panic!("failed to write supplemental header: {e}"));

    // Generate FFI bindings via bindgen.
    // Use the processed cint.h from OUT_DIR so includes resolve correctly.
    let bindings = bindgen::Builder::default()
        .header(suppl_h.to_string_lossy())
        .clang_arg(format!("-I{}", out_dir.display()))
        .clang_arg(format!("-I{}", libcint_root.join("src").display()))
        .allowlist_function("int1e_ovlp_sph|int1e_kin_sph|int1e_nuc_sph|int2e_sph|int2c2e_sph|int3c1e_sph|int3c1e_p2_sph|int3c2e_sph|int3c2e_cart|int4c1e_sph|int4c1e_cart|int1e_ovlp_cart|int1e_kin_cart|int1e_nuc_cart|int2e_cart|int2c2e_cart|int3c1e_cart|int3c1e_p2_cart|int3c2e_ip1_cart|int1e_ovlp_spinor|int1e_kin_spinor|int1e_nuc_spinor|int2e_spinor|int2c2e_spinor|int3c1e_spinor|int3c2e_spinor|int2e_stg_sph|int2e_stg_ip1_sph|int2e_stg_ipip1_sph|int2e_stg_ipvip1_sph|int2e_stg_ip1ip2_sph|int2e_yp_sph|int2e_yp_ip1_sph|int2e_yp_ipip1_sph|int2e_yp_ipvip1_sph|int2e_yp_ip1ip2_sph|int1e_r2_origi_sph|int1e_r4_origi_sph|int1e_r2_origi_ip2_sph|int1e_r4_origi_ip2_sph|int1e_grids_sph|int1e_grids_ip_sph|int1e_grids_ipvip_sph|int1e_grids_spvsp_sph|int1e_grids_ipip_sph|int2e_breit_r1p2_spinor|int2e_breit_r2p2_spinor|int3c1e_r2_origk_sph|int3c1e_r4_origk_sph|int3c1e_r6_origk_sph|int3c1e_ip1_r2_origk_sph|int3c1e_ip1_r4_origk_sph|int3c1e_ip1_r6_origk_sph|int3c2e_sph_ssc|ECPscalar_sph|ECPscalar_cart|ECPscalar_ipnuc_sph|ECPscalar_ipnuc_cart|CINTcgto_spheric|CINTinit_optimizer|CINTdel_optimizer|CINTlen_cart|CINTlen_spinor|CINTcgto_cart|CINTcgto_spinor|CINTtot_pgto_spheric|CINTtot_pgto_spinor|CINTtot_cgto_cart|CINTtot_cgto_spheric|CINTtot_cgto_spinor|CINTshells_cart_offset|CINTshells_spheric_offset|CINTshells_spinor_offset|CINTgto_norm|CINTc2s_bra_sph")
        .allowlist_type("CINTOpt")
        .generate()
        .expect("failed to generate oracle libcint bindings");

    bindings
        .write_to_file(out_dir.join("oracle_bindings.rs"))
        .expect("failed to write oracle libcint bindings");

    // ─────────────────────────────────────────────────────────────────────
    // Phase 19 D-01 (REVISED 2026-05-12): PySCF nr_ecp parallel cc::Build.
    //
    // libcint 6.1.3 upstream ships zero ECP source files (verified in
    // 19-RESEARCH.md §"libcint v6.1.3 upstream — NO ECP CODE PRESENT").
    // The C-level ECP code historically attributed to "libcint ECP"
    // actually lives in PySCF's `pyscf/lib/gto/nr_ecp.{c,h}` +
    // `nr_ecp_deriv.c` (Apache-2.0, same author Qiming Sun). We vendor
    // these into `vendor/pyscf-nr-ecp/` (NOT into libcint-master/) to
    // keep the upstream libcint sync clean.
    //
    // This build emits the `has_vendor_pyscf_nr_ecp` cfg so downstream
    // FFI declarations (added by Plan 03+) can compile-out cleanly on
    // hosts where CINTX_ORACLE_BUILD_VENDOR is not set.
    // ─────────────────────────────────────────────────────────────────────
    let pyscf_root = workspace_root.join("vendor/pyscf-nr-ecp");
    let mut ecp_build = cc::Build::new();
    ecp_build
        .warnings(false)
        // PySCF's nr_ecp.{c,_deriv.c} and the cintx-authored dgemm_shim.c
        // use C99 features (mid-block declarations, for-loop init decls,
        // `_Complex double`). gnu89 (libcint's baseline) is too strict;
        // use gnu99 for the PySCF subtree. The libcint cc::Build above
        // continues to use gnu89 — the two trees are compiled into
        // separate static libraries (`cintx_oracle_vendor` vs
        // `cintx_pyscf_nr_ecp`) so the flag choice does not bleed.
        .flag("-std=gnu99")
        .flag("-Wno-implicit-function-declaration")
        // Processed cint.h + cint_config.h (libcint vendor headers shared
        // with the upstream libcint build above).
        .include(&out_dir)
        .include(libcint_root.join("include"))
        .include(libcint_root.join("src"))
        // PySCF nr_ecp.h declares its slot constants (AS_ECPBAS_OFFSET=18,
        // AS_NECPBAS=19, RADI_POWER=3, SO_TYPE_OF=4, ECP_LMAX=5) and the
        // shim np_helper.h + vhf/fblas.h sit under the same include root.
        .include(pyscf_root.join("include"))
        .file(pyscf_root.join("src/nr_ecp.c"))
        .file(pyscf_root.join("src/nr_ecp_deriv.c"))
        // Minimal cintx-authored dgemm_ reference implementation. Avoids a
        // hard system-BLAS link dependency for the default oracle path; a
        // future cintx-oracle build can drop this file and link `-lblas`
        // instead. See `.planning/notes/pyscf-nr-ecp-vendor-subset.md`.
        .file(pyscf_root.join("src/dgemm_shim.c"))
        .compile("cintx_pyscf_nr_ecp");

    println!("cargo:rustc-link-lib=static=cintx_pyscf_nr_ecp");
    println!("cargo:rustc-cfg=has_vendor_pyscf_nr_ecp");

    for src in &[
        "src/nr_ecp.c",
        "src/nr_ecp_deriv.c",
        "src/dgemm_shim.c",
        "include/nr_ecp.h",
        "include/gto/nr_ecp.h",
        "include/np_helper/np_helper.h",
        "include/vhf/fblas.h",
        "LICENSE",
        "NOTICE",
    ] {
        println!("cargo:rerun-if-changed={}", pyscf_root.join(src).display());
    }
}
