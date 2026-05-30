//! Operator manifest and resolution (stub).

pub mod generated;
pub mod resolver;

#[cfg(test)]
#[test]
fn legacy_wrapper_manifest_matches_misc() {
    use resolver::{HelperKind, Resolver};
    use std::collections::BTreeSet;

    fn macro_kind(base_symbol: &str) -> Option<&'static str> {
        match base_symbol {
            "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e" | "int3c1e" | "int3c1e_p2"
            | "int3c2e_ip1" | "int2e_ip1" => Some("ALL_CINT"),
            // ALL_CINT1E families (libcint 6.1.3 src/autocode/grad1.c + intor1.c) — cart/sph/base
            // legacy wrappers only, no optimizer variants.
            "int1e_kin" | "int1e_ipovlp" | "int1e_ipkin" | "int1e_ipnuc" | "int1e_iprinv" => {
                Some("ALL_CINT1E")
            }
            // int3c2e (plain) has no misc.h wrapper in libcint 6.1.3 upstream — distinct from
            // int3c2e_ip1. int1e_ecp_iprinv (ECP) likewise has no cint* legacy macro wrapper.
            // Returning None keeps this test green without synthetic legacy entries (per Phase 18
            // PATTERNS.md §Step 3 Option A).
            "int3c2e" | "int1e_ecp_iprinv" => None,
            _ => None,
        }
    }

    let mut base_symbols = BTreeSet::new();
    for entry in Resolver::entries_by_kind(HelperKind::Operator) {
        if !entry.compiled_in_profiles.contains(&"base") {
            continue;
        }
        if let Some(base_symbol) = entry.symbol_name.strip_suffix("_cart") {
            base_symbols.insert(base_symbol.to_owned());
        } else if let Some(base_symbol) = entry.symbol_name.strip_suffix("_sph") {
            base_symbols.insert(base_symbol.to_owned());
        } else if let Some(base_symbol) = entry.symbol_name.strip_suffix("_spinor") {
            base_symbols.insert(base_symbol.to_owned());
        }
    }

    let mut expected = BTreeSet::new();
    for base_symbol in base_symbols {
        // Base symbols without misc.h wrappers (e.g. int3c2e plain in libcint 6.1.3)
        // do not contribute legacy entries — skip rather than panic.
        let Some(kind) = macro_kind(&base_symbol) else {
            continue;
        };
        expected.insert(format!("c{base_symbol}_cart"));
        expected.insert(format!("c{base_symbol}_sph"));
        expected.insert(format!("c{base_symbol}"));
        if kind == "ALL_CINT" {
            expected.insert(format!("c{base_symbol}_cart_optimizer"));
            expected.insert(format!("c{base_symbol}_sph_optimizer"));
            expected.insert(format!("c{base_symbol}_optimizer"));
        }
    }

    let actual: BTreeSet<String> = Resolver::entries_by_kind(HelperKind::Legacy)
        .into_iter()
        .filter(|entry| entry.compiled_in_profiles.contains(&"base"))
        .map(|entry| entry.symbol_name.to_owned())
        .collect();

    assert_eq!(
        actual, expected,
        "legacy wrappers drifted from misc.h macro rules"
    );
}
