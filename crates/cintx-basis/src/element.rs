//! Element symbol <-> atomic number mapping (Z = 1..=118).

/// IUPAC symbols indexed by `Z - 1`.
const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Resolve an element symbol (case-insensitive) to its atomic number.
#[must_use]
pub fn atomic_number(symbol: &str) -> Option<u16> {
    let symbol = symbol.trim();
    SYMBOLS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(symbol))
        .map(|index| index as u16 + 1)
}

/// Resolve an atomic number to its element symbol.
#[must_use]
pub fn symbol(atomic_number: u16) -> Option<&'static str> {
    if atomic_number == 0 {
        return None;
    }
    SYMBOLS.get(atomic_number as usize - 1).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_element() {
        for z in 1..=118_u16 {
            let s = symbol(z).expect("symbol should exist");
            assert_eq!(atomic_number(s), Some(z), "round trip failed for Z={z}");
        }
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(atomic_number("fe"), Some(26));
        assert_eq!(atomic_number("FE"), Some(26));
        assert_eq!(atomic_number(" Fe "), Some(26));
    }

    #[test]
    fn rejects_unknown_symbols() {
        assert_eq!(atomic_number("Xx"), None);
        assert_eq!(symbol(0), None);
        assert_eq!(symbol(119), None);
    }
}
