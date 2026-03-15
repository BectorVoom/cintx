use crate::contracts::{IntegralFamily, OperatorKind, Representation};
use crate::errors::LibcintRsError;
use crate::runtime::ExecutionRequest;

use super::ffi::CpuKernelSymbol;
use super::spinor_3c1e::{adapter_route, Spinor3c1eAdapter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuRouteKey {
    pub family: IntegralFamily,
    pub operator: OperatorKind,
    pub representation: Representation,
}

impl CpuRouteKey {
    pub const fn new(
        family: IntegralFamily,
        operator: OperatorKind,
        representation: Representation,
    ) -> Self {
        Self {
            family,
            operator,
            representation,
        }
    }

    pub const fn canonical_family(self) -> &'static str {
        match self.family {
            IntegralFamily::OneElectron => "1e",
            IntegralFamily::TwoElectron => "2e",
            IntegralFamily::TwoCenterTwoElectron => "2c2e",
            IntegralFamily::ThreeCenterOneElectron => "3c1e",
            IntegralFamily::ThreeCenterTwoElectron => "3c2e",
        }
    }

    pub const fn canonical_operator(self) -> &'static str {
        match self.operator {
            OperatorKind::Overlap => "overlap",
            OperatorKind::Kinetic => "kinetic",
            OperatorKind::NuclearAttraction => "nuclear-attraction",
            OperatorKind::ElectronRepulsion => "electron-repulsion",
        }
    }

    pub const fn canonical_representation(self) -> &'static str {
        match self.representation {
            Representation::Cartesian => "cart",
            Representation::Spherical => "sph",
            Representation::Spinor => "spinor",
        }
    }
}

impl From<&ExecutionRequest> for CpuRouteKey {
    fn from(request: &ExecutionRequest) -> Self {
        Self::new(
            request.operator.family,
            request.operator.kind,
            request.representation,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuRouteTarget {
    Direct(CpuKernelSymbol),
    ThreeCenterOneElectronSpinor(Spinor3c1eAdapter),
}

impl CpuRouteTarget {
    pub fn entry_symbol(self) -> CpuKernelSymbol {
        match self {
            Self::Direct(symbol) => symbol,
            Self::ThreeCenterOneElectronSpinor(adapter) => adapter.driver_symbol,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteSurface {
    Safe,
    Raw,
    CAbi,
}

impl RouteSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Raw => "raw",
            Self::CAbi => "capi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteSurfaceGroup {
    Safe,
    Raw,
    CAbi,
    All,
}

impl RouteSurfaceGroup {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Raw => "raw",
            Self::CAbi => "capi",
            Self::All => "all",
        }
    }

    pub const fn supports(self, surface: RouteSurface) -> bool {
        match self {
            Self::All => true,
            Self::Safe => matches!(surface, RouteSurface::Safe),
            Self::Raw => matches!(surface, RouteSurface::Raw),
            Self::CAbi => matches!(surface, RouteSurface::CAbi),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteStability {
    Stable,
    Optional,
    UnstableSource,
}

impl RouteStability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Optional => "optional",
            Self::UnstableSource => "unstable_source",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteKind {
    DirectKernel,
    TransformFromCart,
    ComposedWorkaround,
    UnsupportedPolicy,
}

impl RouteKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectKernel => "direct_kernel",
            Self::TransformFromCart => "transform_from_cart",
            Self::ComposedWorkaround => "composed_workaround",
            Self::UnsupportedPolicy => "unsupported_policy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteOptimizerMode {
    Supported,
    IgnoredButInvariant,
    NotApplicable,
}

impl RouteOptimizerMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::IgnoredButInvariant => "ignored_but_invariant",
            Self::NotApplicable => "not_applicable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteStatus {
    Implemented,
    UnsupportedPolicy,
}

impl RouteStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::UnsupportedPolicy => "unsupported_policy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteEntryKernel {
    Direct(CpuKernelSymbol),
    OneElectronOverlapCartesian,
    OneElectronKineticCartesian,
    ThreeCenterOneElectronSpinorAdapter,
    UnsupportedPolicy,
}

impl RouteEntryKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct(symbol) => symbol.name(),
            Self::OneElectronOverlapCartesian => "cpu::one_e::overlap_cartesian",
            Self::OneElectronKineticCartesian => "cpu::one_e::kinetic_cartesian",
            Self::ThreeCenterOneElectronSpinorAdapter => {
                "cpu::three_center_one_electron_spinor_adapter"
            }
            Self::UnsupportedPolicy => "policy::unsupported",
        }
    }

    pub const fn route_target(self) -> Option<CpuRouteTarget> {
        match self {
            Self::Direct(symbol) => Some(CpuRouteTarget::Direct(symbol)),
            Self::OneElectronOverlapCartesian => {
                Some(CpuRouteTarget::Direct(CpuKernelSymbol::Int1eOvlpCart))
            }
            Self::OneElectronKineticCartesian => {
                Some(CpuRouteTarget::Direct(CpuKernelSymbol::Int1eKinCart))
            }
            Self::ThreeCenterOneElectronSpinorAdapter => {
                Some(CpuRouteTarget::ThreeCenterOneElectronSpinor(adapter_route()))
            }
            Self::UnsupportedPolicy => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuRouteManifestEntry {
    pub route_id: &'static str,
    pub key: CpuRouteKey,
    pub surface_group: RouteSurfaceGroup,
    pub feature_flag: &'static str,
    pub stability: RouteStability,
    pub support_predicate: &'static str,
    pub route_kind: RouteKind,
    pub backend_set: &'static [&'static str],
    pub entry_kernel: RouteEntryKernel,
    pub transform_chain: &'static [&'static str],
    pub writer_contract: &'static str,
    pub optimizer_mode: RouteOptimizerMode,
    pub parity_gate: &'static str,
    pub status: RouteStatus,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedCpuRoute {
    pub route_id: &'static str,
    pub key: CpuRouteKey,
    pub route_target: CpuRouteTarget,
    pub route_kind: RouteKind,
    pub entry_kernel: RouteEntryKernel,
}

const BACKEND_CPU: &[&str] = &["cpu"];
const NO_TRANSFORMS: &[&str] = &[];

const fn implemented_entry(
    route_id: &'static str,
    key: CpuRouteKey,
    route_kind: RouteKind,
    entry_kernel: RouteEntryKernel,
    writer_contract: &'static str,
    optimizer_mode: RouteOptimizerMode,
    parity_gate: &'static str,
    transform_chain: &'static [&'static str],
    notes: &'static str,
) -> CpuRouteManifestEntry {
    CpuRouteManifestEntry {
        route_id,
        key,
        surface_group: RouteSurfaceGroup::All,
        feature_flag: "none",
        stability: RouteStability::Stable,
        support_predicate: "always",
        route_kind,
        backend_set: BACKEND_CPU,
        entry_kernel,
        transform_chain,
        writer_contract,
        optimizer_mode,
        parity_gate,
        status: RouteStatus::Implemented,
        notes,
    }
}

const fn unsupported_entry(
    route_id: &'static str,
    key: CpuRouteKey,
    support_predicate: &'static str,
    notes: &'static str,
) -> CpuRouteManifestEntry {
    CpuRouteManifestEntry {
        route_id,
        key,
        surface_group: RouteSurfaceGroup::All,
        feature_flag: "none",
        stability: RouteStability::Stable,
        support_predicate,
        route_kind: RouteKind::UnsupportedPolicy,
        backend_set: &[],
        entry_kernel: RouteEntryKernel::UnsupportedPolicy,
        transform_chain: NO_TRANSFORMS,
        writer_contract: "n/a",
        optimizer_mode: RouteOptimizerMode::NotApplicable,
        parity_gate: "n/a",
        status: RouteStatus::UnsupportedPolicy,
        notes,
    }
}

static ROUTE_COVERAGE_MANIFEST: [CpuRouteManifestEntry; 23] = [
    implemented_entry(
        "int1e_ovlp.cart.cpu.specialized.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Cartesian,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::OneElectronOverlapCartesian,
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_overlap_cartesian_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed Cartesian overlap parity route",
    ),
    implemented_entry(
        "int1e_ovlp.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Spherical,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSph),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_overlap_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spherical overlap parity route",
    ),
    implemented_entry(
        "int1e_ovlp.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Overlap,
            Representation::Spinor,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSpinor),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_overlap_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spinor overlap parity route",
    ),
    implemented_entry(
        "int1e_kin.cart.cpu.specialized.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Kinetic,
            Representation::Cartesian,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::OneElectronKineticCartesian,
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_kinetic_cartesian_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed Cartesian kinetic parity route",
    ),
    implemented_entry(
        "int1e_kin.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Kinetic,
            Representation::Spherical,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSph),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_kinetic_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spherical kinetic parity route",
    ),
    implemented_entry(
        "int1e_kin.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSpinor),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_kinetic_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spinor kinetic parity route",
    ),
    implemented_entry(
        "int1e_nuc.cart.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::NuclearAttraction,
            Representation::Cartesian,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucCart),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_nuclear_cartesian_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed Cartesian nuclear-attraction parity route",
    ),
    implemented_entry(
        "int1e_nuc.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::NuclearAttraction,
            Representation::Spherical,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSph),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_nuclear_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spherical nuclear-attraction parity route",
    ),
    implemented_entry(
        "int1e_nuc.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::NuclearAttraction,
            Representation::Spinor,
        ),
        RouteKind::ComposedWorkaround,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSpinor),
        "libcint_flat_col_major_1e",
        RouteOptimizerMode::NotApplicable,
        "tests/one_e_nuclear_noncart_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed spinor nuclear-attraction parity route",
    ),
    implemented_entry(
        "int2e_eri.cart.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eCart),
        "libcint_flat_col_major_2e",
        RouteOptimizerMode::Supported,
        "tests/two_e_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed two-electron Cartesian parity route",
    ),
    implemented_entry(
        "int2e_eri.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSph),
        "libcint_flat_col_major_2e",
        RouteOptimizerMode::Supported,
        "tests/two_e_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed two-electron spherical parity route",
    ),
    implemented_entry(
        "int2e_eri.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSpinor),
        "libcint_flat_col_major_2e",
        RouteOptimizerMode::Supported,
        "tests/two_e_wrapper_parity.rs",
        NO_TRANSFORMS,
        "wrapper-backed two-electron spinor parity route",
    ),
    implemented_entry(
        "int2c2e_eri.cart.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2c2eIp1Cart),
        "libcint_flat_col_major_2c2e",
        RouteOptimizerMode::Supported,
        "tests/phase2_cpu_execution_matrix.rs",
        NO_TRANSFORMS,
        "two-center two-electron Cartesian direct deterministic route",
    ),
    implemented_entry(
        "int2c2e_eri.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2c2eIp1Sph),
        "libcint_flat_col_major_2c2e",
        RouteOptimizerMode::Supported,
        "tests/phase2_cpu_execution_matrix.rs",
        NO_TRANSFORMS,
        "two-center two-electron spherical direct deterministic route",
    ),
    implemented_entry(
        "int2c2e_eri.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2c2eIp1Spinor),
        "libcint_flat_col_major_2c2e",
        RouteOptimizerMode::Supported,
        "tests/phase2_cpu_execution_matrix.rs",
        NO_TRANSFORMS,
        "two-center two-electron spinor direct deterministic route",
    ),
    implemented_entry(
        "int3c1e_kin.cart.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Cartesian,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Cart),
        "libcint_flat_col_major_3c",
        RouteOptimizerMode::Supported,
        "tests/three_c_wrapper_parity.rs",
        NO_TRANSFORMS,
        "three-center one-electron Cartesian direct deterministic route",
    ),
    implemented_entry(
        "int3c1e_kin.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spherical,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Sph),
        "libcint_flat_col_major_3c",
        RouteOptimizerMode::Supported,
        "tests/three_c_wrapper_parity.rs",
        NO_TRANSFORMS,
        "three-center one-electron spherical direct deterministic route",
    ),
    unsupported_entry(
        "int3c1e_kin.spinor.policy.unsupported.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
        "unsupported_policy",
        "3c1e spinor kernel aborts in upstream libcint; route remains policy-blocked until Rust-native transform implementation lands",
    ),
    implemented_entry(
        "int3c2e_eri.cart.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Cart),
        "libcint_flat_col_major_3c",
        RouteOptimizerMode::Supported,
        "tests/three_c_wrapper_parity.rs",
        NO_TRANSFORMS,
        "three-center two-electron Cartesian direct deterministic route",
    ),
    implemented_entry(
        "int3c2e_eri.sph.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Sph),
        "libcint_flat_col_major_3c",
        RouteOptimizerMode::Supported,
        "tests/three_c_wrapper_parity.rs",
        NO_TRANSFORMS,
        "three-center two-electron spherical direct deterministic route",
    ),
    implemented_entry(
        "int3c2e_eri.spinor.cpu.direct.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
        ),
        RouteKind::DirectKernel,
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Spinor),
        "libcint_flat_col_major_3c",
        RouteOptimizerMode::Supported,
        "tests/three_c_wrapper_parity.rs",
        NO_TRANSFORMS,
        "three-center two-electron spinor direct deterministic route",
    ),
    unsupported_entry(
        "int2e_ovlp.sph.policy.unsupported.v1",
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::Overlap,
            Representation::Spherical,
        ),
        "unsupported_policy",
        "invalid family/operator pair remains explicitly unsupported by policy",
    ),
    unsupported_entry(
        "int3c2e_kin.spinor.policy.unsupported.v1",
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
        "unsupported_policy",
        "kinetic 3c2e is outside the production denominator",
    ),
];

pub fn route_manifest_entries() -> &'static [CpuRouteManifestEntry] {
    &ROUTE_COVERAGE_MANIFEST
}

pub fn route_manifest_lock_json() -> &'static str {
    include_str!("route_coverage_manifest.lock.json")
}

pub fn resolve_route_request(
    request: &ExecutionRequest,
    surface: RouteSurface,
) -> Result<ResolvedCpuRoute, LibcintRsError> {
    resolve_route(CpuRouteKey::from(request), surface)
}

pub fn resolve_safe_route(request: &ExecutionRequest) -> Result<ResolvedCpuRoute, LibcintRsError> {
    resolve_route_request(request, RouteSurface::Safe)
}

pub fn resolve_raw_route(request: &ExecutionRequest) -> Result<ResolvedCpuRoute, LibcintRsError> {
    resolve_route_request(request, RouteSurface::Raw)
}

pub fn resolve_capi_route(request: &ExecutionRequest) -> Result<ResolvedCpuRoute, LibcintRsError> {
    resolve_route_request(request, RouteSurface::CAbi)
}

pub fn resolve_route(
    key: CpuRouteKey,
    surface: RouteSurface,
) -> Result<ResolvedCpuRoute, LibcintRsError> {
    let mut implemented: Option<CpuRouteManifestEntry> = None;
    let mut matched_policy_entry = false;

    for entry in ROUTE_COVERAGE_MANIFEST {
        if entry.key != key || !entry.surface_group.supports(surface) {
            continue;
        }

        matched_policy_entry = true;
        if entry.status == RouteStatus::Implemented {
            if let Some(previous) = implemented {
                return Err(LibcintRsError::BackendFailure {
                    backend: "cpu.route",
                    detail: format!(
                        "route policy has duplicate implemented entries for {}/{}/{} on surface {}: {} and {}",
                        key.canonical_family(),
                        key.canonical_operator(),
                        key.canonical_representation(),
                        surface.as_str(),
                        previous.route_id,
                        entry.route_id,
                    ),
                });
            }
            implemented = Some(entry);
        }
    }

    if let Some(entry) = implemented {
        let route_target =
            entry
                .entry_kernel
                .route_target()
                .ok_or_else(|| LibcintRsError::BackendFailure {
                    backend: "cpu.route",
                    detail: format!(
                        "implemented route `{}` has non-executable entry kernel `{}`",
                        entry.route_id,
                        entry.entry_kernel.as_str(),
                    ),
                })?;
        return Ok(ResolvedCpuRoute {
            route_id: entry.route_id,
            key: entry.key,
            route_target,
            route_kind: entry.route_kind,
            entry_kernel: entry.entry_kernel,
        });
    }

    if matched_policy_entry {
        return Err(unsupported_policy_error());
    }

    Err(missing_manifest_route_error())
}

pub fn route_request(request: &ExecutionRequest) -> Result<CpuRouteTarget, LibcintRsError> {
    resolve_safe_route(request).map(|resolved| resolved.route_target)
}

pub fn route(key: CpuRouteKey) -> Result<CpuRouteTarget, LibcintRsError> {
    resolve_route(key, RouteSurface::Safe).map(|resolved| resolved.route_target)
}

fn unsupported_policy_error() -> LibcintRsError {
    LibcintRsError::UnsupportedApi {
        api: "cpu.route",
        reason: "route is unsupported by shared route coverage policy",
    }
}

fn missing_manifest_route_error() -> LibcintRsError {
    LibcintRsError::UnsupportedApi {
        api: "cpu.route",
        reason: "route is outside the shared route coverage manifest",
    }
}
