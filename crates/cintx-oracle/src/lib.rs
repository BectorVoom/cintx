//! Oracle adapter for profile-aware parity checks.

pub mod compare;
pub mod fixtures;

#[cfg(has_vendor_libcint)]
pub mod vendor_ffi;

// Profile-aware parity report entrypoints surfaced for xtask/CI gate consumers.
pub use compare::{generate_profile_parity_report, generate_phase2_parity_report, verify_helper_surface_coverage, tolerance_for_family, Phase2ParityReport, FamilyTolerance};
// Profile-aware fixture builders and required profile/family constants for gate wiring.
pub use fixtures::{build_profile_representation_matrix, build_required_profile_matrices, manifest_oracle_families, is_oracle_eligible_family, PHASE4_APPROVED_PROFILES, PHASE4_ORACLE_FAMILIES};

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
