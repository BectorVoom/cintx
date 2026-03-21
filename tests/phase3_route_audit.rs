#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

use std::collections::{BTreeSet, HashSet};

use cintx::{
    CpuRouteKey, ExecutionRequest, IntegralFamily, LibcintRsError, Operator, OperatorKind,
    RawQueryRequest, Representation, RouteStatus, RouteSurface, WorkspaceQueryOptions, raw,
    resolve_capi_route, resolve_raw_route, resolve_route, resolve_safe_route,
    route_manifest_entries, route_manifest_lock_json, safe,
};
use phase2_fixtures::{stable_phase2_matrix, stable_raw_layout, stable_safe_basis};
use serde_json::Value;

#[test]
fn resolver_surface_equivalence_for_implemented_routes() {
    for entry in route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.status == RouteStatus::Implemented)
    {
        let request = request_for_key(entry.key);
        let safe =
            resolve_safe_route(&request).unwrap_or_else(|err| panic!("safe route failed: {err:?}"));
        let raw =
            resolve_raw_route(&request).unwrap_or_else(|err| panic!("raw route failed: {err:?}"));
        let capi =
            resolve_capi_route(&request).unwrap_or_else(|err| panic!("capi route failed: {err:?}"));

        assert_eq!(safe.route_id, entry.route_id);
        assert_eq!(raw.route_id, entry.route_id);
        assert_eq!(capi.route_id, entry.route_id);
        assert_eq!(safe.entry_kernel, raw.entry_kernel);
        assert_eq!(safe.entry_kernel, capi.entry_kernel);
    }
}

#[test]
fn route_coverage_completeness_for_supported_policy_set() {
    let mut expected_stable = stable_phase2_matrix()
        .into_iter()
        .map(|row| row.route_key())
        .collect::<HashSet<_>>();
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::Kinetic,
        Representation::Cartesian,
    ));
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::Kinetic,
        Representation::Spherical,
    ));
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::Kinetic,
        Representation::Spinor,
    ));
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::NuclearAttraction,
        Representation::Cartesian,
    ));
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::NuclearAttraction,
        Representation::Spherical,
    ));
    expected_stable.insert(CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::NuclearAttraction,
        Representation::Spinor,
    ));
    let expected_optional = [
        CpuRouteKey::new(
            IntegralFamily::FourCenterOneElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ),
        CpuRouteKey::new(
            IntegralFamily::FourCenterOneElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ),
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let mut expected = expected_stable.clone();
    expected.extend(expected_optional.iter().copied());

    let implemented = route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.status == RouteStatus::Implemented)
        .map(|entry| entry.key)
        .collect::<HashSet<_>>();

    assert_eq!(implemented, expected);
    assert_eq!(implemented.len(), 22);

    for key in expected_stable {
        resolve_route(key, RouteSurface::Safe).unwrap_or_else(|err| {
            panic!("shared resolver unexpectedly rejected stable route {key:?}: {err:?}")
        });
    }
    for key in expected_optional {
        let request = request_for_key(key);
        resolve_safe_route(&request).unwrap_or_else(|err| {
            panic!("shared resolver unexpectedly rejected optional route {key:?}: {err:?}")
        });
    }
}

#[test]
fn unsupported_routes_fail_through_shared_policy_path() {
    let unsupported = route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.status == RouteStatus::UnsupportedPolicy)
        .collect::<Vec<_>>();
    assert!(
        !unsupported.is_empty(),
        "route manifest must keep explicit unsupported policy rows"
    );

    for entry in unsupported {
        for surface in [RouteSurface::Safe, RouteSurface::Raw, RouteSurface::CAbi] {
            let err = resolve_route(entry.key, surface)
                .expect_err("unsupported policy route should not resolve");
            assert!(
                matches!(
                    err,
                    LibcintRsError::UnsupportedApi {
                        api: "cpu.route",
                        ..
                    }
                ),
                "expected shared policy UnsupportedApi for {surface:?} {:?}, got {err:?}",
                entry.key
            );
        }
    }
}

#[test]
fn supported_query_surfaces_follow_shared_route_policy() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for entry in route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.status == RouteStatus::Implemented)
    {
        let options = route_options_for_key(entry.key);
        let operator = Operator::new(entry.key.family, entry.key.operator)
            .unwrap_or_else(|err| panic!("implemented route operator pair must be valid: {err:?}"));
        let shell_tuple = shell_tuple_for_family(entry.key.family);
        let raw_shls = shell_tuple
            .iter()
            .map(|value| i32::try_from(*value).expect("shell indices must fit i32"))
            .collect::<Vec<_>>();

        let safe_workspace = safe::query_workspace(
            &basis,
            operator,
            entry.key.representation,
            shell_tuple.as_slice(),
            &options,
        )
        .unwrap_or_else(|err| {
            panic!(
                "safe query rejected supported route {:?}: {err:?}",
                entry.key
            )
        });

        let raw_workspace = raw::query_workspace(
            &basis,
            operator,
            entry.key.representation,
            shell_tuple.as_slice(),
            None,
            &options,
        )
        .unwrap_or_else(|err| {
            panic!(
                "raw query rejected supported route {:?}: {err:?}",
                entry.key
            )
        });

        let compat_workspace = raw::query_workspace_compat_with_sentinels(
            operator,
            entry.key.representation,
            RawQueryRequest {
                shls: raw_shls.as_slice(),
                dims: None,
                atm: &atm,
                bas: &bas,
                env: &env,
                out: None,
                cache: None,
                opt: None,
            },
            &options,
        )
        .unwrap_or_else(|err| {
            panic!(
                "raw.compat query rejected supported route {:?}: {err:?}",
                entry.key
            )
        });

        assert_eq!(safe_workspace.dims, raw_workspace.dims);
        assert_eq!(safe_workspace.dims, compat_workspace.dims);
    }
}

#[test]
fn unsupported_query_surfaces_fail_through_shared_policy_path() {
    let basis = stable_safe_basis();
    let (atm, bas, env) = stable_raw_layout();

    for entry in route_manifest_entries()
        .iter()
        .copied()
        .filter(|entry| entry.status == RouteStatus::UnsupportedPolicy)
    {
        let options = route_options_for_key(entry.key);
        let operator = match Operator::new(entry.key.family, entry.key.operator) {
            Ok(operator) => operator,
            // Invalid family/operator pairs are blocked by typed operator construction
            // before query/evaluate surfaces can invoke routing.
            Err(_) => continue,
        };
        let shell_tuple = shell_tuple_for_family(entry.key.family);
        let raw_shls = shell_tuple
            .iter()
            .map(|value| i32::try_from(*value).expect("shell indices must fit i32"))
            .collect::<Vec<_>>();

        let safe_err = safe::query_workspace(
            &basis,
            operator,
            entry.key.representation,
            shell_tuple.as_slice(),
            &options,
        )
        .expect_err("safe query should fail for unsupported policy route");
        assert!(
            matches!(
                safe_err.error,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "safe query did not fail through shared policy for {:?}: {safe_err:?}",
            entry.key
        );

        let raw_err = raw::query_workspace(
            &basis,
            operator,
            entry.key.representation,
            shell_tuple.as_slice(),
            None,
            &options,
        )
        .expect_err("raw query should fail for unsupported policy route");
        assert!(
            matches!(
                raw_err.error,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "raw query did not fail through shared policy for {:?}: {raw_err:?}",
            entry.key
        );

        let compat_err = raw::query_workspace_compat_with_sentinels(
            operator,
            entry.key.representation,
            RawQueryRequest {
                shls: raw_shls.as_slice(),
                dims: None,
                atm: &atm,
                bas: &bas,
                env: &env,
                out: None,
                cache: None,
                opt: None,
            },
            &options,
        )
        .expect_err("raw.compat query should fail for unsupported policy route");
        assert!(
            matches!(
                compat_err.error,
                LibcintRsError::UnsupportedApi {
                    api: "cpu.route",
                    ..
                }
            ),
            "raw.compat query did not fail through shared policy for {:?}: {compat_err:?}",
            entry.key
        );
    }
}

#[test]
fn bypass_detection_in_entry_layers() {
    let targets = [
        (
            "src/runtime/executor.rs",
            &[
                "fill_safe_one_e_overlap_cartesian",
                "fill_safe_one_e_kinetic_cartesian",
                "uses_safe_overlap_cartesian",
                "uses_safe_kinetic_cartesian",
            ][..],
        ),
        (
            "src/runtime/raw/evaluate.rs",
            &[
                "fill_raw_one_e_overlap_cartesian",
                "fill_raw_one_e_kinetic_cartesian",
                "uses_raw_overlap_cartesian",
                "uses_raw_kinetic_cartesian",
                "CpuKernelSymbol::Int1eKinCart",
            ][..],
        ),
        (
            "src/api/raw.rs",
            &[
                "fill_raw_one_e_overlap_cartesian",
                "fill_raw_one_e_kinetic_cartesian",
                "CpuKernelSymbol::Int1eKinCart",
                "CpuRouteTarget::Direct",
            ][..],
        ),
        (
            "src/api/safe.rs",
            &[
                "fill_safe_one_e_overlap_cartesian",
                "fill_safe_one_e_kinetic_cartesian",
                "CpuKernelSymbol::Int1eKinCart",
                "CpuRouteTarget::Direct",
            ][..],
        ),
    ];

    for (path, forbidden) in targets {
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
        for needle in forbidden {
            assert!(
                !source.contains(needle),
                "{path} still contains forbidden routing bypass marker `{needle}`"
            );
        }
    }
}

#[test]
fn route_manifest_lock_matches_resolver_table() {
    let lock: Value =
        serde_json::from_str(route_manifest_lock_json()).expect("route lock JSON must parse");
    assert_eq!(lock["schema_version"].as_u64(), Some(1));

    let lock_entries = lock["entries"]
        .as_array()
        .expect("route lock entries must be an array");

    let from_lock = lock_entries
        .iter()
        .map(route_tuple_from_lock)
        .collect::<BTreeSet<_>>();
    let from_resolver = route_manifest_entries()
        .iter()
        .map(route_tuple_from_resolver)
        .collect::<BTreeSet<_>>();

    assert_eq!(from_lock, from_resolver);
    assert_eq!(from_lock.len(), route_manifest_entries().len());
}

fn request_for_key(key: CpuRouteKey) -> ExecutionRequest {
    let shell_tuple = shell_tuple_for_family(key.family);

    let operator = Operator::new(key.family, key.operator)
        .unwrap_or_else(|err| panic!("implemented route operator pair should be valid: {err:?}"));

    ExecutionRequest::from_safe(
        operator,
        key.representation,
        shell_tuple.as_slice(),
        &route_options_for_key(key),
    )
}

fn shell_tuple_for_family(family: IntegralFamily) -> Vec<usize> {
    match family {
        IntegralFamily::OneElectron | IntegralFamily::TwoCenterTwoElectron => vec![0usize, 1usize],
        IntegralFamily::ThreeCenterOneElectron | IntegralFamily::ThreeCenterTwoElectron => {
            vec![0usize, 1usize, 2usize]
        }
        IntegralFamily::TwoElectron | IntegralFamily::FourCenterOneElectron => {
            vec![0usize, 1usize, 2usize, 3usize]
        }
    }
}

fn route_options() -> WorkspaceQueryOptions {
    WorkspaceQueryOptions {
        memory_limit_bytes: None,
        backend_candidate: "cpu",
        feature_flags: vec!["phase3-route-audit"],
    }
}

fn route_options_for_key(key: CpuRouteKey) -> WorkspaceQueryOptions {
    let mut options = route_options();
    if key.family == IntegralFamily::FourCenterOneElectron {
        options.feature_flags.push("with-4c1e");
    }
    options
}

fn route_tuple_from_lock(entry: &Value) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        required_str(entry, "route_id"),
        required_str(entry, "canonical_family"),
        required_str(entry, "operator"),
        required_str(entry, "representation"),
        required_str(entry, "surface_group"),
        required_str(entry, "route_kind"),
        required_str(entry, "entry_kernel"),
        required_str(entry, "status"),
    )
}

fn route_tuple_from_resolver(entry: &cintx::CpuRouteManifestEntry) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        entry.route_id,
        entry.key.canonical_family(),
        entry.key.canonical_operator(),
        entry.key.canonical_representation(),
        entry.surface_group.as_str(),
        entry.route_kind.as_str(),
        entry.entry_kernel.as_str(),
        entry.status.as_str(),
    )
}

fn required_str<'a>(entry: &'a Value, field: &str) -> &'a str {
    entry[field]
        .as_str()
        .unwrap_or_else(|| panic!("route lock field `{field}` must be a string"))
}
