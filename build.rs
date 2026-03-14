use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LIBCINT_ROOT: &str = "libcint-master";
const LIBCINT_VERSION: &str = "6.1.3";
const LIBCINT_SOVERSION: &str = "6";

const REQUIRED_CPU_SOURCES: &[&str] = &[
    "src/c2f.c",
    "src/cart2sph.c",
    "src/cint1e.c",
    "src/cint2e.c",
    "src/cint_bas.c",
    "src/fblas.c",
    "src/g1e.c",
    "src/g2e.c",
    "src/misc.c",
    "src/optimizer.c",
    "src/fmt.c",
    "src/rys_wheeler.c",
    "src/eigh.c",
    "src/rys_roots.c",
    "src/find_roots.c",
    "src/cint2c2e.c",
    "src/g2c2e.c",
    "src/cint3c2e.c",
    "src/g3c2e.c",
    "src/cint3c1e.c",
    "src/g3c1e.c",
    "src/breit.c",
    "src/cint1e_a.c",
    "src/cint3c1e_a.c",
    "src/cint1e_grids.c",
    "src/g1e_grids.c",
    "src/autocode/breit1.c",
    "src/autocode/dkb.c",
    "src/autocode/gaunt1.c",
    "src/autocode/grad1.c",
    "src/autocode/grad2.c",
    "src/autocode/hess.c",
    "src/autocode/int3c1e.c",
    "src/autocode/int3c2e.c",
    "src/autocode/intor1.c",
    "src/autocode/intor2.c",
    "src/autocode/intor3.c",
    "src/autocode/intor4.c",
    "src/autocode/deriv3.c",
    "src/autocode/int1e_grids1.c",
    "src/autocode/deriv4.c",
    "src/autocode/lresc.c",
];

#[derive(Debug, Clone, Copy)]
enum BuildDiagnosticKind {
    MissingSource,
    MissingTemplate,
    IoFailure,
}

impl BuildDiagnosticKind {
    const fn as_code(self) -> &'static str {
        match self {
            Self::MissingSource => "missing-source",
            Self::MissingTemplate => "missing-template",
            Self::IoFailure => "io-failure",
        }
    }
}

fn fail(kind: BuildDiagnosticKind, message: impl AsRef<str>) -> ! {
    panic!("[cintx-build:{}] {}", kind.as_code(), message.as_ref());
}

fn read_required(path: &Path, kind: BuildDiagnosticKind) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        fail(
            kind,
            format!("could not read `{}`: {error}", path.display()),
        )
    })
}

fn write_generated(path: &Path, content: &str) {
    fs::write(path, content).unwrap_or_else(|error| {
        fail(
            BuildDiagnosticKind::IoFailure,
            format!("could not write `{}`: {error}", path.display()),
        )
    });
}

fn replace_cmakedefines(template: &str, replacements: &[(&str, &str)]) -> String {
    let mut output = String::with_capacity(template.len() + 128);
    for line in template.lines() {
        let mut replacement = line;
        for (needle, value) in replacements {
            if line.trim_start() == *needle {
                replacement = value;
                break;
            }
        }
        output.push_str(replacement);
        output.push('\n');
    }
    output
}

fn render_cint_header(template: &str) -> String {
    let versioned = template
        .replace("@cint_VERSION@", LIBCINT_VERSION)
        .replace("@cint_SOVERSION@", LIBCINT_SOVERSION);
    replace_cmakedefines(
        &versioned,
        &[
            ("#cmakedefine I8", "/* #undef I8 */"),
            ("#cmakedefine CACHE_SIZE_I8", "/* #undef CACHE_SIZE_I8 */"),
        ],
    )
}

fn render_config_header(template: &str) -> String {
    replace_cmakedefines(
        template,
        &[
            ("#cmakedefine HAVE_EXPL", "#define HAVE_EXPL 1"),
            ("#cmakedefine HAVE_SQRTL", "#define HAVE_SQRTL 1"),
            ("#cmakedefine HAVE_FABSL", "#define HAVE_FABSL 1"),
            (
                "#cmakedefine HAVE_QUADMATH_H",
                "/* #undef HAVE_QUADMATH_H */",
            ),
            (
                "#cmakedefine WITH_RANGE_COULOMB",
                "/* #undef WITH_RANGE_COULOMB */",
            ),
        ],
    )
}

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|error| fail(BuildDiagnosticKind::IoFailure, error.to_string())),
    );
    let libcint_root = manifest_dir.join(LIBCINT_ROOT);
    if !libcint_root.is_dir() {
        fail(
            BuildDiagnosticKind::MissingSource,
            format!(
                "vendored libcint directory is missing: expected `{}`",
                libcint_root.display()
            ),
        );
    }

    let cint_h_template = libcint_root.join("include/cint.h.in");
    if !cint_h_template.is_file() {
        fail(
            BuildDiagnosticKind::MissingTemplate,
            format!("required template missing: `{}`", cint_h_template.display()),
        );
    }

    let cint_config_template = libcint_root.join("src/cint_config.h.in");
    if !cint_config_template.is_file() {
        fail(
            BuildDiagnosticKind::MissingTemplate,
            format!(
                "required template missing: `{}`",
                cint_config_template.display()
            ),
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", cint_h_template.display());
    println!("cargo:rerun-if-changed={}", cint_config_template.display());

    let out_dir = PathBuf::from(
        env::var("OUT_DIR")
            .unwrap_or_else(|error| fail(BuildDiagnosticKind::IoFailure, error.to_string())),
    );
    let generated_include_dir = out_dir.join("libcint-generated/include");
    let generated_src_dir = out_dir.join("libcint-generated/src");

    fs::create_dir_all(&generated_include_dir).unwrap_or_else(|error| {
        fail(
            BuildDiagnosticKind::IoFailure,
            format!(
                "could not create generated include directory `{}`: {error}",
                generated_include_dir.display()
            ),
        )
    });
    fs::create_dir_all(&generated_src_dir).unwrap_or_else(|error| {
        fail(
            BuildDiagnosticKind::IoFailure,
            format!(
                "could not create generated source directory `{}`: {error}",
                generated_src_dir.display()
            ),
        )
    });

    let rendered_cint_h = render_cint_header(&read_required(
        &cint_h_template,
        BuildDiagnosticKind::MissingTemplate,
    ));
    let rendered_cint_config_h = render_config_header(&read_required(
        &cint_config_template,
        BuildDiagnosticKind::MissingTemplate,
    ));

    write_generated(&generated_include_dir.join("cint.h"), &rendered_cint_h);
    write_generated(
        &generated_src_dir.join("cint_config.h"),
        &rendered_cint_config_h,
    );

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .define("WITH_FORTRAN", None)
        .define("WITH_CINT2_INTERFACE", None)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-fno-math-errno")
        .include(&generated_include_dir)
        .include(&generated_src_dir)
        .include(libcint_root.join("include"))
        .include(libcint_root.join("src"));

    for source in REQUIRED_CPU_SOURCES {
        let path = libcint_root.join(source);
        if !path.is_file() {
            fail(
                BuildDiagnosticKind::MissingSource,
                format!("required source file missing: `{}`", path.display()),
            );
        }
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(path);
    }

    build.compile("cint_phase2_cpu");
    println!("cargo:rustc-link-lib=m");
}
