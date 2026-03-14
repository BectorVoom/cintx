use cintx::{
    CanonicalSymbolIdentity, CompiledManifestLock, IntegralFamily, LockUpdateApproval,
    LockUpdateReason, ManifestGovernanceError, ManifestLockEntry, ManifestProfile, OperatorKind,
    Representation, StabilityClass,
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

#[test]
fn manifest_profile_union_is_stable() {
    let canonicalized_entry = ManifestLockEntry::from_profile_labels(
        CanonicalSymbolIdentity::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Cartesian,
            " INT1E Ovlp-CART ",
        )
        .expect("symbol identity should canonicalize"),
        [
            "with_4c1e",
            "BASE",
            "f12",
            "with-4c1e + with-f12",
            "with-f12",
        ],
        StabilityClass::Stable,
    )
    .expect("entry should parse profile aliases");

    let stable_lock = CompiledManifestLock::new(vec![canonicalized_entry]).expect("lock is valid");
    assert_eq!(stable_lock.entries[0].id.symbol, "int1e_ovlp_cart");
    assert_eq!(
        stable_lock.entries[0].profiles,
        vec![
            ManifestProfile::Base,
            ManifestProfile::WithF12,
            ManifestProfile::With4c1e,
            ManifestProfile::WithF12With4c1e,
        ],
    );
    stable_lock
        .validate_profile_union()
        .expect("phase-3 profile union must stay fixed");

    let drift_lock = CompiledManifestLock::new(vec![
        ManifestLockEntry::new(
            CanonicalSymbolIdentity::new(
                IntegralFamily::TwoElectron,
                OperatorKind::ElectronRepulsion,
                Representation::Spherical,
                "int2e_sph",
            )
            .expect("symbol identity should be valid"),
            [ManifestProfile::Base, ManifestProfile::WithF12],
            StabilityClass::Stable,
        )
        .expect("entry should be valid"),
    ])
    .expect("lock is valid");
    let drift = drift_lock
        .validate_profile_union()
        .expect_err("missing governed profile combinations must be detected");
    assert!(matches!(
        drift,
        ManifestGovernanceError::ProfileUnionDrift { .. }
    ));
}

#[test]
fn lock_drift_requires_explicit_update() {
    let baseline_lock = CompiledManifestLock::new(manifest_entries_covering_all_profiles())
        .expect("baseline lock should be valid");

    let mut drifted_entries = manifest_entries_covering_all_profiles();
    drifted_entries[0].stability = StabilityClass::Experimental;
    let drifted_lock =
        CompiledManifestLock::new(drifted_entries).expect("drifted lock should still be valid");

    let unapproved = drifted_lock
        .enforce_drift_policy(&baseline_lock, None)
        .expect_err("lock drift must fail without explicit approval");
    assert!(matches!(
        unapproved,
        ManifestGovernanceError::UnapprovedLockDrift { .. }
    ));

    let empty_rationale_approval =
        LockUpdateApproval::new(LockUpdateReason::ProfilePolicyChange, "   ");
    let empty_rationale = drifted_lock
        .enforce_drift_policy(&baseline_lock, Some(&empty_rationale_approval))
        .expect_err("approval rationale cannot be blank");
    assert!(matches!(
        empty_rationale,
        ManifestGovernanceError::EmptyApprovalRationale
    ));

    let approved = LockUpdateApproval::new(
        LockUpdateReason::ProfilePolicyChange,
        "phase-3 profile policy changed with governance review",
    );
    drifted_lock
        .enforce_drift_policy(&baseline_lock, Some(&approved))
        .expect("approved lock drift should pass");

    baseline_lock
        .enforce_drift_policy(&baseline_lock, None)
        .expect("identical locks should not require approval");
}
