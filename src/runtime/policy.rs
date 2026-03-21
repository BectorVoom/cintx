use crate::contracts::{IntegralFamily, Operator, Representation};
use crate::errors::LibcintRsError;
use crate::manifest::canonicalize_profile_label;

pub(crate) const WITH_4C1E_FEATURE: &str = "with-4c1e";
pub(crate) const OUTSIDE_VALIDATED_4C1E_REASON: &str =
    "route is outside the Validated4C1E policy envelope";

pub(crate) fn feature_flag_enabled<'a>(
    flags: impl IntoIterator<Item = &'a str>,
    required: &str,
) -> bool {
    flags
        .into_iter()
        .map(canonicalize_profile_label)
        .any(|flag| flag == required)
}

pub(crate) fn enforce_optional_family_policy<'a>(
    operator: Operator,
    representation: Representation,
    shell_angular_momentum: &[u8],
    natural_dims: &[usize],
    dims: &[usize],
    backend_candidate: &str,
    feature_flags: impl IntoIterator<Item = &'a str>,
) -> Result<(), LibcintRsError> {
    if operator.family() != IntegralFamily::FourCenterOneElectron {
        return Ok(());
    }

    if !feature_flag_enabled(feature_flags, WITH_4C1E_FEATURE) {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: "route is unsupported by shared route coverage policy",
        });
    }
    if backend_candidate != "cpu" {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: OUTSIDE_VALIDATED_4C1E_REASON,
        });
    }
    if !matches!(
        representation,
        Representation::Cartesian | Representation::Spherical
    ) {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: OUTSIDE_VALIDATED_4C1E_REASON,
        });
    }
    if dims != natural_dims {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: OUTSIDE_VALIDATED_4C1E_REASON,
        });
    }
    if shell_angular_momentum.iter().any(|&l| l > 4) {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.route",
            reason: OUTSIDE_VALIDATED_4C1E_REASON,
        });
    }

    Ok(())
}
