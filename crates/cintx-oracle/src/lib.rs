//! Oracle adapter for profile-aware parity checks.

pub mod compare;
pub mod fixtures;

pub use fixtures::{build_profile_representation_matrix, build_required_profile_matrices};

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
