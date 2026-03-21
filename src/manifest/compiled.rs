use crate::contracts::{IntegralFamily, OperatorKind, Representation};
use crate::manifest::lock::{
    CanonicalSymbolIdentity, CompiledManifestLock, ManifestGovernanceError, ManifestLockEntry,
    ManifestProfile, StabilityClass,
};
use crate::runtime::backend::cpu::{
    ALL_BOUND_SYMBOLS, CpuRouteManifestEntry, RouteEntryKernel, RouteStability, RouteStatus,
    route_manifest_entries,
};

pub const COMPILED_MANIFEST_LOCK_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/compiled_manifest.lock.json");

pub fn compiled_manifest_lock_json() -> &'static str {
    include_str!("../../compiled_manifest.lock.json")
}

pub fn parse_compiled_manifest_lock_json(
    raw: &str,
) -> Result<CompiledManifestLock, ManifestGovernanceError> {
    let lock: CompiledManifestLock = serde_json::from_str(raw)?;
    lock.validate_schema_invariants()?;
    lock.validate_profile_union()?;
    Ok(lock)
}

pub fn generated_compiled_manifest_lock() -> Result<CompiledManifestLock, ManifestGovernanceError> {
    let entries = route_manifest_entries()
        .iter()
        .filter(|entry| entry.status == RouteStatus::Implemented)
        .map(compiled_entry_from_route)
        .collect::<Result<Vec<_>, _>>()?;

    CompiledManifestLock::new(entries)
}

pub fn audit_compiled_manifest_lock(
    lock: &CompiledManifestLock,
) -> Result<(), ManifestGovernanceError> {
    lock.validate_schema_invariants()?;
    lock.validate_profile_union()?;
    audit_route_manifest_invariants()?;

    let generated = generated_compiled_manifest_lock()?;
    generated.enforce_drift_policy(lock, None)?;
    Ok(())
}

fn compiled_entry_from_route(
    entry: &CpuRouteManifestEntry,
) -> Result<ManifestLockEntry, ManifestGovernanceError> {
    let symbol = compiled_symbol_for_route(entry)?;
    let id = CanonicalSymbolIdentity::new(
        entry.key.family,
        entry.key.operator,
        entry.key.representation,
        symbol,
    )?;
    let profiles = profiles_for_feature_flag(entry.feature_flag)?;
    ManifestLockEntry::new(id, profiles, stability_for_route(entry.stability))
}

fn compiled_symbol_for_route(
    entry: &CpuRouteManifestEntry,
) -> Result<&'static str, ManifestGovernanceError> {
    match entry.entry_kernel {
        RouteEntryKernel::Direct(symbol) => Ok(symbol.name()),
        RouteEntryKernel::OneElectronOverlapCartesian => Ok("int1e_ovlp_cart"),
        RouteEntryKernel::OneElectronKineticCartesian => Ok("int1e_kin_cart"),
        RouteEntryKernel::ThreeCenterOneElectronSpinorAdapter => Ok("int3c1e_p2_spinor"),
        RouteEntryKernel::UnsupportedPolicy => Err(ManifestGovernanceError::AuditInvariant {
            detail: format!(
                "implemented route `{}` resolved to unsupported entry kernel",
                entry.route_id
            ),
        }),
    }
}

fn profiles_for_feature_flag(
    feature_flag: &str,
) -> Result<Vec<ManifestProfile>, ManifestGovernanceError> {
    match feature_flag {
        "none" => Ok(ManifestProfile::approved_scope()),
        "with-f12" => Ok(vec![
            ManifestProfile::WithF12,
            ManifestProfile::WithF12With4c1e,
        ]),
        "with-4c1e" => Ok(vec![
            ManifestProfile::With4c1e,
            ManifestProfile::WithF12With4c1e,
        ]),
        "with-f12+with-4c1e" | "with-4c1e+with-f12" => Ok(vec![ManifestProfile::WithF12With4c1e]),
        _ => Err(ManifestGovernanceError::UnknownFeatureFlag {
            feature_flag: feature_flag.to_string(),
        }),
    }
}

fn stability_for_route(stability: RouteStability) -> StabilityClass {
    match stability {
        RouteStability::Stable => StabilityClass::Stable,
        RouteStability::Optional | RouteStability::UnstableSource => StabilityClass::Experimental,
    }
}

fn audit_route_manifest_invariants() -> Result<(), ManifestGovernanceError> {
    let bound_symbols = ALL_BOUND_SYMBOLS
        .iter()
        .map(|symbol| symbol.name())
        .collect::<std::collections::BTreeSet<_>>();
    let implemented = route_manifest_entries()
        .iter()
        .filter(|entry| entry.status == RouteStatus::Implemented);

    for entry in implemented {
        if entry.entry_kernel.route_target().is_none() {
            return Err(ManifestGovernanceError::AuditInvariant {
                detail: format!(
                    "implemented route `{}` has no callable backend target",
                    entry.route_id
                ),
            });
        }

        if let RouteEntryKernel::Direct(symbol) = entry.entry_kernel {
            if !bound_symbols.contains(symbol.name()) {
                return Err(ManifestGovernanceError::AuditInvariant {
                    detail: format!(
                        "implemented route `{}` references direct kernel `{}` that is not bound",
                        entry.route_id,
                        symbol.name()
                    ),
                });
            }
        }

        let is_4c1e = entry.key.family == IntegralFamily::FourCenterOneElectron;
        if is_4c1e {
            if entry.feature_flag != "with-4c1e" || entry.stability != RouteStability::Optional {
                return Err(ManifestGovernanceError::AuditInvariant {
                    detail: format!(
                        "4c1e route `{}` must be optional and gated by `with-4c1e`",
                        entry.route_id
                    ),
                });
            }
        } else if entry.feature_flag == "with-4c1e" {
            return Err(ManifestGovernanceError::AuditInvariant {
                detail: format!(
                    "non-4c1e route `{}` unexpectedly uses `with-4c1e` gate",
                    entry.route_id
                ),
            });
        }

        if entry.feature_flag.contains("with-f12")
            && (entry.key.family != IntegralFamily::TwoElectron
                || entry.key.operator != OperatorKind::ElectronRepulsion
                || entry.key.representation != Representation::Spherical)
        {
            return Err(ManifestGovernanceError::AuditInvariant {
                detail: format!(
                    "F12-gated route `{}` must be 2e/electron-repulsion/spherical",
                    entry.route_id
                ),
            });
        }
    }

    Ok(())
}
