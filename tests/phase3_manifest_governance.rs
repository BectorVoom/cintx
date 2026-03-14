use cintx::{
    CanonicalSymbolIdentity, CompiledManifestLock, IntegralFamily, ManifestLockEntry,
    ManifestProfile, OperatorKind, Representation, StabilityClass,
};

fn manifest_entries_covering_all_profiles() -> Vec<ManifestLockEntry> {
    let base = ManifestLockEntry::new(
        CanonicalSymbolIdentity::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Cartesian,
            "INT1E-OVLP-CART",
        )
        .expect("symbol identity should canonicalize"),
        [ManifestProfile::Base],
        StabilityClass::Stable,
    )
    .expect("entry should be valid");

    let with_f12 = ManifestLockEntry::new(
        CanonicalSymbolIdentity::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
            "int2e_sph",
        )
        .expect("symbol identity should be valid"),
        [ManifestProfile::WithF12],
        StabilityClass::Experimental,
    )
    .expect("entry should be valid");

    let with_4c1e = ManifestLockEntry::new(
        CanonicalSymbolIdentity::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
            "int3c1e_p2_spinor",
        )
        .expect("symbol identity should be valid"),
        [ManifestProfile::With4c1e],
        StabilityClass::Stable,
    )
    .expect("entry should be valid");

    let with_f12_and_4c1e = ManifestLockEntry::new(
        CanonicalSymbolIdentity::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
            "int3c2e_ip1_spinor",
        )
        .expect("symbol identity should be valid"),
        [ManifestProfile::WithF12With4c1e],
        StabilityClass::Deprecated,
    )
    .expect("entry should be valid");

    vec![base, with_f12, with_4c1e, with_f12_and_4c1e]
}

#[test]
fn manifest_schema_invariants() {
    let lock = CompiledManifestLock::new(manifest_entries_covering_all_profiles())
        .expect("typed schema should build for valid entries");

    assert_eq!(lock.schema_version, 1);
    assert_eq!(lock.entries.len(), 4);
    assert_eq!(lock.entries[0].id.symbol, "int1e_ovlp_cart");
    assert_eq!(
        lock.profile_scope.approved,
        vec![
            ManifestProfile::Base,
            ManifestProfile::WithF12,
            ManifestProfile::With4c1e,
            ManifestProfile::WithF12With4c1e,
        ],
    );
    for entry in &lock.entries {
        assert!(
            !entry.profiles.is_empty(),
            "every lock entry needs explicit profile membership"
        );
    }
}
