//! Oracle adapter for Phase 2 parity checks.

pub mod compare;
pub mod fixtures;

#[cfg(test)]
mod tests {
    #[test]
    fn exports_and_compat_raw_edge_compile() {
        #[allow(unused_imports)]
        use super::{compare, fixtures};
        #[allow(unused_imports)]
        use cintx_compat::raw;
    }
}
