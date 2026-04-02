/// CubeCL wgpu backend capability snapshot and reason taxonomy.
///
/// This module implements D-02 (explicit typed failures), D-04 (adapter capability
/// context for reproducibility), and D-12 (specific unsupported reason taxonomy).

/// A snapshot of the concrete wgpu adapter capabilities captured at bootstrap time.
///
/// The snapshot is deterministically hashable — callers can produce a `capability_fingerprint`
/// that changes when any field relevant to execution changes, enabling drift detection per D-04.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuCapabilitySnapshot {
    /// Human-readable adapter name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub adapter_name: String,
    /// Backend API string (e.g. `"vulkan"`, `"dx12"`, `"metal"`, `"gl"`, `"webgpu"`).
    pub backend_api: String,
    /// Device type classification (e.g. `"discrete"`, `"integrated"`, `"virtual"`, `"cpu"`, `"other"`).
    pub device_type: String,
    /// Vendor PCI ID as decimal string.
    pub vendor_id: String,
    /// Device PCI ID as decimal string.
    pub device_id: String,
    /// Sorted list of feature flag names that are relevant for integral kernel execution.
    pub features: Vec<String>,
    /// Sorted list of `"key:value"` pairs for limits relevant to compute.
    pub limits: Vec<String>,
}

impl WgpuCapabilitySnapshot {
    /// Construct a snapshot from raw adapter fields.
    pub fn new(
        adapter_name: impl Into<String>,
        backend_api: impl Into<String>,
        device_type: impl Into<String>,
        vendor_id: u32,
        device_id: u32,
        mut features: Vec<String>,
        mut limits: Vec<String>,
    ) -> Self {
        features.sort();
        limits.sort();
        Self {
            adapter_name: adapter_name.into(),
            backend_api: backend_api.into(),
            device_type: device_type.into(),
            vendor_id: vendor_id.to_string(),
            device_id: device_id.to_string(),
            features,
            limits,
        }
    }
}

/// Compute a deterministic capability fingerprint for a [`WgpuCapabilitySnapshot`].
///
/// The fingerprint is a 64-bit FNV-1a hash of the snapshot's fields in a stable,
/// deterministic order.  It changes whenever any field that affects execution
/// changes, enabling D-04 reproducibility checks.
///
/// # Stability guarantee
/// - Same inputs always produce the same fingerprint on the same platform.
/// - Any change to adapter name, backend API, device type, vendor/device ID,
///   feature list, or limits list will produce a different fingerprint.
pub fn capability_fingerprint(snapshot: &WgpuCapabilitySnapshot) -> u64 {
    // FNV-1a 64-bit constants.
    const OFFSET_BASIS: u64 = 14695981039346656037_u64;
    const FNV_PRIME: u64 = 1099511628211_u64;

    let mut hash = OFFSET_BASIS;

    let feed = |hash: &mut u64, data: &[u8]| {
        for &byte in data {
            *hash ^= byte as u64;
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Field separator to prevent prefix collisions.
        *hash ^= b'\0' as u64;
        *hash = hash.wrapping_mul(FNV_PRIME);
    };

    feed(&mut hash, snapshot.adapter_name.as_bytes());
    feed(&mut hash, snapshot.backend_api.as_bytes());
    feed(&mut hash, snapshot.device_type.as_bytes());
    feed(&mut hash, snapshot.vendor_id.as_bytes());
    feed(&mut hash, snapshot.device_id.as_bytes());

    for feature in &snapshot.features {
        feed(&mut hash, feature.as_bytes());
    }
    // Feature list boundary separator.
    feed(&mut hash, b"||features||");

    for limit in &snapshot.limits {
        feed(&mut hash, limit.as_bytes());
    }
    feed(&mut hash, b"||limits||");

    hash
}

/// Typed reason taxonomy for capability and unsupported-scope failures.
///
/// Matches D-02 (explicit capability failures) and D-12 (specific unsupported
/// reason taxonomy).  Each variant maps to a reason-prefixed diagnostic string
/// via [`CapabilityReason::to_reason_string`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityReason {
    /// No wgpu adapter was found matching the requested selector.
    MissingAdapter,
    /// A required device feature is missing on the selected adapter.
    MissingFeature(String),
    /// A required device limit is below the minimum required for correct execution.
    ///
    /// Fields: `(limit_name, actual_value, required_value)`.
    LimitTooLow(String, u64, u64),
    /// The requested integral family is not supported by this backend.
    FamilyUnsupported(String),
    /// The requested basis representation is not supported by this backend.
    RepresentationUnsupported(String),
}

impl CapabilityReason {
    /// Emit a reason-prefixed diagnostic string suitable for inclusion in
    /// `UnsupportedApi { requested }` error messages.
    ///
    /// Format:
    /// - `missing_adapter`
    /// - `missing_feature:<name>`
    /// - `limit_too_low:<name>:<actual>/<required>`
    /// - `family_unsupported:<family>`
    /// - `representation_unsupported:<repr>`
    pub fn to_reason_string(&self) -> String {
        match self {
            CapabilityReason::MissingAdapter => "missing_adapter".to_owned(),
            CapabilityReason::MissingFeature(name) => format!("missing_feature:{name}"),
            CapabilityReason::LimitTooLow(name, actual, required) => {
                format!("limit_too_low:{name}:{actual}/{required}")
            }
            CapabilityReason::FamilyUnsupported(family) => {
                format!("family_unsupported:{family}")
            }
            CapabilityReason::RepresentationUnsupported(repr) => {
                format!("representation_unsupported:{repr}")
            }
        }
    }
}

/// The output of a wgpu adapter preflight check.
///
/// Combines a capability snapshot with the adapter's preflight outcome
/// so callers can decide whether to proceed (all required capabilities
/// satisfied) or fail closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuPreflightReport {
    /// Capability snapshot from the selected adapter.
    pub snapshot: WgpuCapabilitySnapshot,
    /// Fingerprint of the snapshot, suitable for storing in a [`cintx_runtime::BackendCapabilityToken`].
    pub fingerprint: u64,
    /// Any capability reasons that indicate the adapter cannot satisfy preflight.
    /// Empty when all required capabilities are present.
    pub unsatisfied: Vec<CapabilityReason>,
}

impl WgpuPreflightReport {
    /// Construct a preflight report from a snapshot and any unsatisfied capability reasons.
    pub fn new(snapshot: WgpuCapabilitySnapshot, unsatisfied: Vec<CapabilityReason>) -> Self {
        let fingerprint = capability_fingerprint(&snapshot);
        Self {
            snapshot,
            fingerprint,
            unsatisfied,
        }
    }

    /// Returns `true` if all required capabilities are satisfied.
    pub fn is_capable(&self) -> bool {
        self.unsatisfied.is_empty()
    }

    /// Returns the first unsatisfied reason, if any.
    pub fn first_reason(&self) -> Option<&CapabilityReason> {
        self.unsatisfied.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> WgpuCapabilitySnapshot {
        WgpuCapabilitySnapshot::new(
            "Test GPU",
            "vulkan",
            "discrete",
            0x10DE_u32,
            0x2684_u32,
            vec!["TIMESTAMP_QUERY".to_owned(), "SUBGROUP".to_owned()],
            vec![
                "max_compute_workgroups_per_dimension:65535".to_owned(),
                "max_storage_buffers_per_shader_stage:8".to_owned(),
            ],
        )
    }

    #[test]
    fn capability_fingerprint_is_deterministic() {
        let snap = sample_snapshot();
        let fp1 = capability_fingerprint(&snap);
        let fp2 = capability_fingerprint(&snap);
        assert_eq!(
            fp1, fp2,
            "Fingerprint must be identical for the same snapshot"
        );

        // Cloned snapshot must produce the same fingerprint.
        let snap2 = snap.clone();
        let fp3 = capability_fingerprint(&snap2);
        assert_eq!(fp1, fp3, "Cloned snapshot must produce the same fingerprint");
    }

    #[test]
    fn capability_fingerprint_changes_when_snapshot_changes() {
        let snap = sample_snapshot();
        let fp_base = capability_fingerprint(&snap);

        // Change adapter name.
        let mut snap2 = snap.clone();
        snap2.adapter_name = "Different GPU".to_owned();
        assert_ne!(
            capability_fingerprint(&snap2),
            fp_base,
            "Fingerprint must change when adapter name changes"
        );

        // Change backend API.
        let mut snap3 = snap.clone();
        snap3.backend_api = "dx12".to_owned();
        assert_ne!(
            capability_fingerprint(&snap3),
            fp_base,
            "Fingerprint must change when backend API changes"
        );

        // Change features list.
        let mut snap4 = snap.clone();
        snap4.features.push("CONSERVATIVE_RASTERIZATION".to_owned());
        snap4.features.sort();
        assert_ne!(
            capability_fingerprint(&snap4),
            fp_base,
            "Fingerprint must change when features change"
        );

        // Change limits list.
        let mut snap5 = snap.clone();
        snap5.limits.push("max_bind_groups:8".to_owned());
        snap5.limits.sort();
        assert_ne!(
            capability_fingerprint(&snap5),
            fp_base,
            "Fingerprint must change when limits change"
        );

        // Change vendor ID.
        let mut snap6 = snap.clone();
        snap6.vendor_id = "0".to_owned();
        assert_ne!(
            capability_fingerprint(&snap6),
            fp_base,
            "Fingerprint must change when vendor ID changes"
        );
    }

    #[test]
    fn unsupported_reason_taxonomy_formats_explicit_reason_classes() {
        // missing_adapter
        let r = CapabilityReason::MissingAdapter;
        assert_eq!(r.to_reason_string(), "missing_adapter");

        // missing_feature:<name>
        let r = CapabilityReason::MissingFeature("TIMESTAMP_QUERY".to_owned());
        let s = r.to_reason_string();
        assert!(
            s.starts_with("missing_feature:"),
            "Should start with 'missing_feature:': {s}"
        );
        assert!(s.contains("TIMESTAMP_QUERY"), "Should contain feature name: {s}");

        // limit_too_low:<name>:<actual>/<required>
        let r = CapabilityReason::LimitTooLow("max_storage_buffers".to_owned(), 4, 8);
        let s = r.to_reason_string();
        assert!(
            s.starts_with("limit_too_low:"),
            "Should start with 'limit_too_low:': {s}"
        );
        assert!(s.contains("max_storage_buffers"), "Should contain limit name: {s}");
        assert!(s.contains("4"), "Should contain actual value: {s}");
        assert!(s.contains("8"), "Should contain required value: {s}");

        // family_unsupported:<family>
        let r = CapabilityReason::FamilyUnsupported("4c1e".to_owned());
        let s = r.to_reason_string();
        assert!(
            s.starts_with("family_unsupported:"),
            "Should start with 'family_unsupported:': {s}"
        );
        assert!(s.contains("4c1e"), "Should contain family name: {s}");

        // representation_unsupported:<repr>
        let r = CapabilityReason::RepresentationUnsupported("spinor".to_owned());
        let s = r.to_reason_string();
        assert!(
            s.starts_with("representation_unsupported:"),
            "Should start with 'representation_unsupported:': {s}"
        );
        assert!(s.contains("spinor"), "Should contain representation name: {s}");
    }

    #[test]
    fn preflight_report_is_capable_when_no_unsatisfied_reasons() {
        let snap = sample_snapshot();
        let report = WgpuPreflightReport::new(snap, vec![]);
        assert!(report.is_capable());
        assert!(report.first_reason().is_none());
        assert_ne!(report.fingerprint, 0, "Fingerprint must be non-zero");
    }

    #[test]
    fn preflight_report_not_capable_when_unsatisfied_reasons_present() {
        let snap = sample_snapshot();
        let report = WgpuPreflightReport::new(
            snap,
            vec![CapabilityReason::MissingFeature("TIMESTAMP_QUERY".to_owned())],
        );
        assert!(!report.is_capable());
        assert!(report.first_reason().is_some());
        assert!(matches!(
            report.first_reason().unwrap(),
            CapabilityReason::MissingFeature(_)
        ));
    }
}
