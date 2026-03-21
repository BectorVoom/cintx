use cintx::{
    FamilyTag, IntegralFamily, ManifestGovernanceError, ManifestProfile, RouteStability,
    RouteStatus, StabilityClass, audit_compiled_manifest_lock, compiled_manifest_lock_json,
    generated_compiled_manifest_lock, parse_compiled_manifest_lock_json, route_manifest_entries,
};

#[test]
fn compiled_manifest_lock_matches_generated_snapshot() {
    let committed = parse_compiled_manifest_lock_json(compiled_manifest_lock_json())
        .expect("compiled manifest lock must parse");
    let generated =
        generated_compiled_manifest_lock().expect("compiled manifest generation must succeed");

    assert_eq!(
        committed
            .canonical_json()
            .expect("committed lock must canonicalize"),
        generated
            .canonical_json()
            .expect("generated lock must canonicalize"),
        "compiled manifest lock is stale; regenerate with `cargo run --bin manifest_audit -- generate`",
    );
    audit_compiled_manifest_lock(&committed)
        .expect("compiled manifest audit must pass for committed lock");
}

#[test]
fn compiled_manifest_audit_detects_stale_lock_drift() {
    let mut drifted =
        generated_compiled_manifest_lock().expect("compiled manifest generation must succeed");
    assert!(
        !drifted.entries.is_empty(),
        "generated compiled manifest cannot be empty",
    );
    drifted.entries[0].stability = StabilityClass::Deprecated;

    let drift = audit_compiled_manifest_lock(&drifted)
        .expect_err("drifted compiled manifest must fail stale-lock audit");
    assert!(matches!(
        drift,
        ManifestGovernanceError::UnapprovedLockDrift { .. }
    ));
}

#[test]
fn optional_family_profile_gates_are_machine_checkable() {
    let implemented = route_manifest_entries()
        .iter()
        .filter(|entry| entry.status == RouteStatus::Implemented)
        .collect::<Vec<_>>();

    let four_c_routes = implemented
        .iter()
        .filter(|entry| entry.key.family == IntegralFamily::FourCenterOneElectron)
        .collect::<Vec<_>>();
    assert_eq!(
        four_c_routes.len(),
        2,
        "4c1e must expose exactly two implemented real-valued routes",
    );
    for entry in four_c_routes {
        assert_eq!(entry.feature_flag, "with-4c1e");
        assert_eq!(entry.stability, RouteStability::Optional);
    }

    let f12_routes = implemented
        .iter()
        .filter(|entry| entry.feature_flag.contains("with-f12"))
        .collect::<Vec<_>>();
    assert!(
        f12_routes.is_empty(),
        "F12 routes must remain absent from implemented stable/optional routing until backend support exists",
    );

    let lock = parse_compiled_manifest_lock_json(compiled_manifest_lock_json())
        .expect("compiled manifest lock must parse");
    let four_c_lock_entries = lock
        .entries
        .iter()
        .filter(|entry| entry.id.family == FamilyTag::FourCenterOneElectron)
        .collect::<Vec<_>>();
    assert_eq!(
        four_c_lock_entries.len(),
        2,
        "compiled manifest must track exactly two 4c1e implemented symbols",
    );
    for entry in four_c_lock_entries {
        assert!(
            !entry.profiles.contains(&ManifestProfile::Base),
            "4c1e entries cannot be visible in base profile",
        );
        assert!(
            !entry.profiles.contains(&ManifestProfile::WithF12),
            "4c1e entries cannot be visible in with-f12 profile without with-4c1e gate",
        );
        assert!(entry.profiles.contains(&ManifestProfile::With4c1e));
        assert!(entry.profiles.contains(&ManifestProfile::WithF12With4c1e));
    }
}
