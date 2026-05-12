//! Effective Core Potential typed surface.
//!
//! Phase 19 Plan 03: replaces Plan 01 stub; field set sourced from
//! 19-PATTERNS.md ecp.rs Phase 19 deviations.
//!
//! `EcpShell` is the typed analog of one PySCF `ecpbas` row. It carries
//! everything the kernel needs to evaluate one ECP shell — atom site,
//! angular momentum channel (local Type-1 sentinel or projected Type-2
//! l-value), radial power, primitive/contraction counts, spin-orbit
//! marker, and the `Arc<[f64]>`-backed exponent + coefficient slabs that
//! match `Shell` for ordinary AO shells.
//!
//! The PySCF C side stores these in a flat i32 slab (`ecpbas`) where
//! slot 3 is reinterpreted as `RADI_POWER` and slot 4 as `SO_TYPE_OF`
//! (vs `NCTR_OF` / `KAPPA_OF` for ordinary `bas` rows). The typed view
//! here is what callers see; the `cintx-compat` boundary packs into the
//! raw layout.
//!
//! See:
//!   - `vendor/pyscf-nr-ecp/include/nr_ecp.h` — `AS_ECPBAS_OFFSET = 18`,
//!     `AS_NECPBAS = 19`, `RADI_POWER = 3`, `SO_TYPE_OF = 4`,
//!     `ECP_LMAX = 5`
//!   - `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-CONTEXT.md`
//!     D-03 / D-04 for the field-set decisions.

use std::sync::Arc;

use crate::error::{CoreError, CoreResult};

/// Maximum projected angular momentum supported by the ECP code path.
///
/// Mirrors PySCF `nr_ecp.h` `ECP_LMAX = 5`. Validated by `EcpShell::try_new`
/// when constructing `EcpChannel::Projected(l)` shells.
pub const ECP_LMAX: u8 = 5;

/// Distinguishes the local Type-1 channel from semi-local Type-2 projector
/// channels.
///
/// PySCF `nr_ecp.h` uses `ANG_OF = -1` as a sentinel for the local channel
/// in the flat `ecpbas` slab. This enum is the typed surface that callers
/// see; the sentinel value is packed back into the raw row at the
/// `cintx-compat` boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcpChannel {
    /// Local Type-1 (Coulomb-like) channel — PySCF convention `ANG_OF = -1`.
    Local,
    /// Semi-local Type-2 projector for angular momentum `l` (`l <= ECP_LMAX`).
    Projected(u8),
}

/// Effective Core Potential shell — typed analog of one PySCF `ecpbas` row.
///
/// Field set is verbatim from 19-PATTERNS.md §"crates/cintx-core/src/ecp.rs"
/// Phase 19 deviations. The `radial_power` field is signed (`i16`) because
/// general ECP forms support negative powers; `so_type` is the PySCF
/// spin-orbit marker (0 = scalar; nonzero is out of scope per Phase 19 D-12).
#[derive(Clone, Debug, PartialEq)]
pub struct EcpShell {
    pub atom_index: u32,
    pub channel: EcpChannel,
    pub radial_power: i16,
    pub nprim: u16,
    pub nctr: u16,
    pub so_type: i16,
    pub exponents: Arc<[f64]>,
    pub coefficients: Arc<[f64]>,
}

impl EcpShell {
    /// Try to build an ECP shell while validating primitive/contraction
    /// counts and finiteness of `exponents`/`coefficients`.
    ///
    /// Invariants enforced (mirrors `Shell::try_new`):
    ///   - `nprim > 0`, `nctr > 0`
    ///   - `exponents.len() == nprim`
    ///   - `coefficients.len() == nprim * nctr`
    ///   - every value in `exponents` and `coefficients` is finite
    ///   - if `channel == EcpChannel::Projected(l)`, then `l <= ECP_LMAX`
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        atom_index: u32,
        channel: EcpChannel,
        radial_power: i16,
        nprim: u16,
        nctr: u16,
        so_type: i16,
        exponents: Arc<[f64]>,
        coefficients: Arc<[f64]>,
    ) -> CoreResult<Self> {
        let nprim_usize = nprim as usize;
        let nctr_usize = nctr as usize;

        if nprim_usize == 0 || nctr_usize == 0 {
            return Err(CoreError::InvalidShellCounts {
                nprim: nprim_usize,
                nctr: nctr_usize,
            });
        }

        if exponents.len() != nprim_usize {
            return Err(CoreError::ShellPrimitiveMismatch {
                field: "exponents",
                expected: nprim_usize,
                actual: exponents.len(),
            });
        }

        let expected_coeffs =
            nprim_usize
                .checked_mul(nctr_usize)
                .ok_or(CoreError::InvalidShellCounts {
                    nprim: nprim_usize,
                    nctr: nctr_usize,
                })?;
        if coefficients.len() != expected_coeffs {
            return Err(CoreError::ShellPrimitiveMismatch {
                field: "coefficients",
                expected: expected_coeffs,
                actual: coefficients.len(),
            });
        }

        if !exponents.iter().all(|value| value.is_finite())
            || !coefficients.iter().all(|value| value.is_finite())
        {
            return Err(CoreError::InvalidNuclearDetail);
        }

        if let EcpChannel::Projected(l) = channel {
            if l > ECP_LMAX {
                return Err(CoreError::EcpAngularMomentumTooHigh {
                    requested: l,
                    max: ECP_LMAX,
                });
            }
        }

        Ok(Self {
            atom_index,
            channel,
            radial_power,
            nprim,
            nctr,
            so_type,
            exponents,
            coefficients,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreError;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_owned().into_boxed_slice())
    }

    #[test]
    fn try_new_validates_field_invariants() {
        let shell = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            2,
            1,
            0,
            arc_f64(&[1.0, 2.0]),
            arc_f64(&[0.5, 0.5]),
        )
        .expect("valid local ECP shell should succeed");
        assert_eq!(shell.atom_index, 0);
        assert_eq!(shell.channel, EcpChannel::Local);
        assert_eq!(shell.radial_power, 0);
        assert_eq!(shell.nprim, 2);
        assert_eq!(shell.nctr, 1);
        assert_eq!(shell.so_type, 0);
        assert_eq!(shell.exponents.len(), 2);
        assert_eq!(shell.coefficients.len(), 2);
    }

    #[test]
    fn try_new_accepts_projected_channel_at_lmax() {
        // l == ECP_LMAX (5) is allowed.
        let shell = EcpShell::try_new(
            0,
            EcpChannel::Projected(ECP_LMAX),
            1,
            1,
            1,
            0,
            arc_f64(&[0.5]),
            arc_f64(&[1.0]),
        )
        .expect("Projected(ECP_LMAX) should succeed");
        assert!(matches!(shell.channel, EcpChannel::Projected(l) if l == ECP_LMAX));
    }

    #[test]
    fn exponent_length_mismatch_is_rejected() {
        // nprim = 2 but only 1 exponent supplied.
        let err = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            2,
            1,
            0,
            arc_f64(&[1.0]),
            arc_f64(&[0.5, 0.5]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::ShellPrimitiveMismatch {
                field: "exponents",
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn coefficient_length_mismatch_is_rejected() {
        // nprim*nctr = 2 but only 1 coefficient supplied.
        let err = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            1,
            2,
            0,
            arc_f64(&[1.0]),
            arc_f64(&[0.5]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::ShellPrimitiveMismatch {
                field: "coefficients",
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn non_finite_exponent_is_rejected() {
        let err = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            2,
            1,
            0,
            arc_f64(&[1.0, f64::NAN]),
            arc_f64(&[0.5, 0.5]),
        )
        .unwrap_err();

        assert!(matches!(err, CoreError::InvalidNuclearDetail));
    }

    #[test]
    fn non_finite_coefficient_is_rejected() {
        let err = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            1,
            1,
            0,
            arc_f64(&[1.0]),
            arc_f64(&[f64::INFINITY]),
        )
        .unwrap_err();

        assert!(matches!(err, CoreError::InvalidNuclearDetail));
    }

    #[test]
    fn zero_nprim_is_rejected() {
        let err = EcpShell::try_new(
            0,
            EcpChannel::Local,
            0,
            0,
            1,
            0,
            arc_f64(&[]),
            arc_f64(&[]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::InvalidShellCounts {
                nprim: 0,
                nctr: 1,
            }
        ));
    }

    #[test]
    fn projected_channel_above_lmax_is_rejected() {
        // l == 6 > ECP_LMAX (5) — must fail.
        let err = EcpShell::try_new(
            0,
            EcpChannel::Projected(6),
            1,
            1,
            1,
            0,
            arc_f64(&[0.5]),
            arc_f64(&[1.0]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::EcpAngularMomentumTooHigh {
                requested: 6,
                max: 5,
            }
        ));
    }
}
