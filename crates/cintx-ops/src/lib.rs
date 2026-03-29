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
            "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e" | "int3c1e_p2" | "int3c2e_ip1" => {
                Some("ALL_CINT")
            }
            "int1e_kin" => Some("ALL_CINT1E"),
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
        let kind = macro_kind(&base_symbol).unwrap_or_else(|| {
            panic!("missing misc.h wrapper macro classification for {base_symbol}")
        });
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
