use crate::layout::{CompatDims, ensure_cache_len};
use crate::optimizer::RawOptimizerHandle;
use cintx_core::{
    Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple, cintxRsError,
};
use cintx_cubecl::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
use cintx_ops::resolver::{HelperKind, OperatorDescriptor, Resolver, ResolverError};
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan, GridsEnvParams,
    HostWorkspaceAllocator, WorkspaceAllocator, WorkspaceQuery, query_workspace, schedule_chunks,
};
use std::mem::size_of;
use std::sync::Arc;

pub const CHARGE_OF: usize = 0;
pub const PTR_COORD: usize = 1;
pub const NUC_MOD_OF: usize = 2;
pub const PTR_ZETA: usize = 3;
pub const PTR_FRAC_CHARGE: usize = 4;
pub const ATM_SLOTS: usize = 6;

pub const ATOM_OF: usize = 0;
pub const ANG_OF: usize = 1;
pub const NPRIM_OF: usize = 2;
pub const NCTR_OF: usize = 3;
pub const KAPPA_OF: usize = 4;
pub const PTR_EXP: usize = 5;
pub const PTR_COEFF: usize = 6;
pub const BAS_SLOTS: usize = 8;

/// First usable index in the env array for user data (coordinates, exponents, coefficients).
///
/// libcint reserves env[0..PTR_ENV_START] for global parameters:
///   PTR_EXPCUTOFF = 0, PTR_COMMON_ORIG = 1..3, PTR_RINV_ORIG = 4..6,
///   PTR_RINV_ZETA = 7, PTR_RANGE_OMEGA = 8, PTR_F12_ZETA = 9, PTR_GTG_ZETA = 10,
///   PTR_GRIDS = 12..19.
///
/// User data (atom coordinates, exponents, coefficients) MUST start at env[20] or later.
/// Placing user data at env[0..19] corrupts the global parameter fields and causes
/// incorrect results for 2e+ integrals that read PTR_RANGE_OMEGA or PTR_EXPCUTOFF.
pub const PTR_ENV_START: usize = 20;

/// Index range in the libcint env array for the common (gauge) origin (x, y, z).
///
/// libcint defines `PTR_COMMON_ORIG = 1` (three consecutive slots 1, 2, 3).
/// Raw callers set `env[1..4] = [x, y, z]` (in Bohr) as the gauge origin for
/// moment / GIAO families. Unset (zero) reads as the default origin `[0,0,0]`;
/// consumers use `common_orig.unwrap_or([0.0; 3])`. This constant is the start
/// index; the origin occupies `PTR_COMMON_ORIG`, `+1`, `+2`.
pub const PTR_COMMON_ORIG: usize = 1;

/// Index range in the libcint env array for the rinv origin (x, y, z).
///
/// libcint defines `PTR_RINV_ORIG = 4` (three consecutive slots 4, 5, 6).
/// Raw callers set `env[4..7] = [x, y, z]` (in Bohr) before calling any iprinv
/// integral. This constant is the start index; the full origin occupies slots
/// `PTR_RINV_ORIG`, `PTR_RINV_ORIG + 1`, `PTR_RINV_ORIG + 2`.
pub const PTR_RINV_ORIG: usize = 4;

/// Index of the F12/STG/YP zeta parameter in the libcint env array.
///
/// libcint defines `PTR_F12_ZETA = 9` in `cint_bas.h`. Raw callers set `env[9] = zeta`
/// before calling any F12/STG/YP integral. This constant allows raw compat code to
/// extract the zeta value from the env array without hardcoding the magic index.
pub const PTR_F12_ZETA: usize = 9;

/// Number of grid points for grids-family integrals.
/// Stored at env[11] by convention (libcint NGRIDS = 11).
pub const NGRIDS: usize = 11;

/// Start index for grid coordinate data in env array.
/// Grid coordinates are packed as env[PTR_GRIDS..PTR_GRIDS + 3*ngrids] (libcint PTR_GRIDS = 12).
pub const PTR_GRIDS: usize = 12;

// =====================================================================
// Phase 19 D-05: ECP slot constants (from PySCF nr_ecp.h, upstream names
// kept verbatim). ecpbas rows reuse the existing BAS_SLOTS = 8 row width;
// slots 3 and 4 are reinterpreted (no separate ecpbas-width constant —
// per the Phase 19 RESEARCH §"ecpbas row width" decision).
// Source: vendor/pyscf-nr-ecp/include/nr_ecp.h
// =====================================================================

/// ECP basis row slot: angular momentum power of `r` in `V_l(r)`.
///
/// Matches PySCF `nr_ecp.h` `RADI_POWER`. Reinterprets the BAS slot index 3
/// (which is `NCTR_OF` for ordinary bas rows). ecpbas rows reuse the existing
/// `BAS_SLOTS = 8` row width.
pub const RADI_POWER: usize = 3;

/// ECP basis row slot: spin-orbit channel marker.
///
/// Matches PySCF `nr_ecp.h` `SO_TYPE_OF`. Reinterprets BAS slot index 4
/// (which is `KAPPA_OF` for ordinary bas rows). 0 = scalar; nonzero = SO
/// (out of scope per Phase 19 D-12).
pub const SO_TYPE_OF: usize = 4;

/// env slot index pointing at the start of the `ecpbas` array.
///
/// Matches PySCF `nr_ecp.h` `AS_ECPBAS_OFFSET = 18`. The kernel-side
/// dispatch reads the ecpbas slice starting at this env index when an
/// `int1e_ecp_*` operator is selected.
pub const AS_ECPBAS_OFFSET: usize = 18;

/// env slot index holding the number of `ecpbas` rows.
///
/// Matches PySCF `nr_ecp.h` `AS_NECPBAS = 19`. `eval_raw` rejects ECP
/// dispatch with a typed `cintxRsError::InvalidEnvParam` when
/// `env[AS_NECPBAS] <= 0` or non-finite.
pub const AS_NECPBAS: usize = 19;

/// Max angular momentum supported by the ECP code path.
///
/// Matches PySCF `nr_ecp.h` `ECP_LMAX = 5`. Mirrored on the typed surface
/// at `cintx_core::ecp::ECP_LMAX`.
pub const ECP_LMAX: usize = 5;

pub const POINT_NUC: i32 = 1;
pub const GAUSSIAN_NUC: i32 = 2;
pub const FRAC_CHARGE_NUC: i32 = 3;
const VALIDATED_4C1E_REASON: &str = "outside Validated4C1E";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawApiId {
    Symbol(&'static str),
}

impl RawApiId {
    pub const INT1E_OVLP_CART: Self = Self::Symbol("int1e_ovlp_cart");
    pub const INT1E_OVLP_SPH: Self = Self::Symbol("int1e_ovlp_sph");
    pub const INT1E_OVLP_SPINOR: Self = Self::Symbol("int1e_ovlp_spinor");

    pub const INT1E_KIN_CART: Self = Self::Symbol("int1e_kin_cart");
    pub const INT1E_KIN_SPH: Self = Self::Symbol("int1e_kin_sph");
    pub const INT1E_KIN_SPINOR: Self = Self::Symbol("int1e_kin_spinor");

    pub const INT1E_NUC_CART: Self = Self::Symbol("int1e_nuc_cart");
    pub const INT1E_NUC_SPH: Self = Self::Symbol("int1e_nuc_sph");
    pub const INT1E_NUC_SPINOR: Self = Self::Symbol("int1e_nuc_spinor");

    pub const INT2E_CART: Self = Self::Symbol("int2e_cart");
    pub const INT2E_SPH: Self = Self::Symbol("int2e_sph");
    pub const INT2E_SPINOR: Self = Self::Symbol("int2e_spinor");

    pub const INT2C2E_CART: Self = Self::Symbol("int2c2e_cart");
    pub const INT2C2E_SPH: Self = Self::Symbol("int2c2e_sph");
    pub const INT2C2E_SPINOR: Self = Self::Symbol("int2c2e_spinor");

    pub const INT3C1E_CART: Self = Self::Symbol("int3c1e_cart");
    pub const INT3C1E_SPH: Self = Self::Symbol("int3c1e_sph");
    pub const INT3C1E_SPINOR: Self = Self::Symbol("int3c1e_spinor");

    pub const INT3C1E_P2_CART: Self = Self::Symbol("int3c1e_p2_cart");
    pub const INT3C1E_P2_SPH: Self = Self::Symbol("int3c1e_p2_sph");
    pub const INT3C1E_P2_SPINOR: Self = Self::Symbol("int3c1e_p2_spinor");

    pub const INT3C2E_IP1_CART: Self = Self::Symbol("int3c2e_ip1_cart");
    pub const INT3C2E_IP1_SPH: Self = Self::Symbol("int3c2e_ip1_sph");
    pub const INT3C2E_IP1_SPINOR: Self = Self::Symbol("int3c2e_ip1_spinor");

    pub const INT4C1E_CART: Self = Self::Symbol("int4c1e_cart");
    pub const INT4C1E_SPH: Self = Self::Symbol("int4c1e_sph");

    // Phase 19 D-05: ECP raw-API symbols. Spinor forms intentionally
    // omitted (out of scope per D-12).
    pub const INT1E_ECP_CART: Self = Self::Symbol("int1e_ecp_cart");
    pub const INT1E_ECP_SPH: Self = Self::Symbol("int1e_ecp_sph");
    pub const INT1E_ECP_IPNUC_CART: Self = Self::Symbol("int1e_ecp_ipnuc_cart");
    pub const INT1E_ECP_IPNUC_SPH: Self = Self::Symbol("int1e_ecp_ipnuc_sph");

    // Phase 21 gradient families.
    pub const INT1E_IPOVLP_CART: Self = Self::Symbol("int1e_ipovlp_cart");
    pub const INT1E_IPOVLP_SPH: Self = Self::Symbol("int1e_ipovlp_sph");
    pub const INT1E_IPOVLP_SPINOR: Self = Self::Symbol("int1e_ipovlp_spinor");

    pub const INT1E_IPKIN_CART: Self = Self::Symbol("int1e_ipkin_cart");
    pub const INT1E_IPKIN_SPH: Self = Self::Symbol("int1e_ipkin_sph");
    pub const INT1E_IPKIN_SPINOR: Self = Self::Symbol("int1e_ipkin_spinor");

    pub const INT1E_IPNUC_CART: Self = Self::Symbol("int1e_ipnuc_cart");
    pub const INT1E_IPNUC_SPH: Self = Self::Symbol("int1e_ipnuc_sph");
    pub const INT1E_IPNUC_SPINOR: Self = Self::Symbol("int1e_ipnuc_spinor");

    pub const INT1E_IPRINV_CART: Self = Self::Symbol("int1e_iprinv_cart");
    pub const INT1E_IPRINV_SPH: Self = Self::Symbol("int1e_iprinv_sph");
    pub const INT1E_IPRINV_SPINOR: Self = Self::Symbol("int1e_iprinv_spinor");

    // Phase 24 Cluster B (MOM-04): plain single-center 1/r Coulomb potential.
    // `int1e_rinv` (rank 1) is the int1e_nuc Rys kernel evaluated at the
    // PTR_RINV_ORIG slot (env[4..6], D-04/OQ-1 correction — NOT PTR_COMMON_ORIG),
    // with charge=+1 and NO atom-sum. `int1e_drinv` (rank 3) is its gradient wrt
    // the rinv center C (= D_I + D_J of the rinv G-tensor). Spinor forms registered
    // for surface completeness; the kernel returns UnsupportedApi (D-09).
    pub const INT1E_RINV_CART: Self = Self::Symbol("int1e_rinv_cart");
    pub const INT1E_RINV_SPH: Self = Self::Symbol("int1e_rinv_sph");
    pub const INT1E_RINV_SPINOR: Self = Self::Symbol("int1e_rinv_spinor");

    pub const INT1E_DRINV_CART: Self = Self::Symbol("int1e_drinv_cart");
    pub const INT1E_DRINV_SPH: Self = Self::Symbol("int1e_drinv_sph");
    pub const INT1E_DRINV_SPINOR: Self = Self::Symbol("int1e_drinv_spinor");

    // Phase 24 Cluster C/D (MOM-04) symbol declarations. The p4 (∇⁴, rank 1) and
    // irp (i·r×∇, rank 9) FAMILIES are now FULLY registered: manifest rows,
    // on-device `#[cube]` kernels, vendor FFI wrappers, and vendor parity tests
    // are all in place. A p4/irp dispatch resolves to a real kernel and is
    // byte-checked against libcint. Spinor forms are registered for surface
    // completeness; the kernel returns UnsupportedApi (D-09).
    pub const INT1E_P4_CART: Self = Self::Symbol("int1e_p4_cart");
    pub const INT1E_P4_SPH: Self = Self::Symbol("int1e_p4_sph");
    pub const INT1E_P4_SPINOR: Self = Self::Symbol("int1e_p4_spinor");

    pub const INT1E_IRP_CART: Self = Self::Symbol("int1e_irp_cart");
    pub const INT1E_IRP_SPH: Self = Self::Symbol("int1e_irp_sph");
    pub const INT1E_IRP_SPINOR: Self = Self::Symbol("int1e_irp_spinor");

    // Phase 24 Cluster A (MOM-01/02/03): overlap-derived position-tensor moment
    // families. Each `_origj` variant is its OWN operator/symbol (D-02): the shared
    // moment kernel branches on origin-source (env[PTR_COMMON_ORIG] for the base
    // family, ket basis center rj for `_origj`). Spinor forms registered for surface
    // completeness; the kernel returns UnsupportedApi (D-09). Symbol strings MUST
    // exactly match the manifest lock entries.
    pub const INT1E_R_CART: Self = Self::Symbol("int1e_r_cart");
    pub const INT1E_R_SPH: Self = Self::Symbol("int1e_r_sph");
    pub const INT1E_R_SPINOR: Self = Self::Symbol("int1e_r_spinor");

    pub const INT1E_Z_CART: Self = Self::Symbol("int1e_z_cart");
    pub const INT1E_Z_SPH: Self = Self::Symbol("int1e_z_sph");
    pub const INT1E_Z_SPINOR: Self = Self::Symbol("int1e_z_spinor");

    pub const INT1E_ZZ_CART: Self = Self::Symbol("int1e_zz_cart");
    pub const INT1E_ZZ_SPH: Self = Self::Symbol("int1e_zz_sph");
    pub const INT1E_ZZ_SPINOR: Self = Self::Symbol("int1e_zz_spinor");

    pub const INT1E_R_ORIGJ_CART: Self = Self::Symbol("int1e_r_origj_cart");
    pub const INT1E_R_ORIGJ_SPH: Self = Self::Symbol("int1e_r_origj_sph");
    pub const INT1E_R_ORIGJ_SPINOR: Self = Self::Symbol("int1e_r_origj_spinor");

    pub const INT1E_Z_ORIGJ_CART: Self = Self::Symbol("int1e_z_origj_cart");
    pub const INT1E_Z_ORIGJ_SPH: Self = Self::Symbol("int1e_z_origj_sph");
    pub const INT1E_Z_ORIGJ_SPINOR: Self = Self::Symbol("int1e_z_origj_spinor");

    pub const INT1E_ZZ_ORIGJ_CART: Self = Self::Symbol("int1e_zz_origj_cart");
    pub const INT1E_ZZ_ORIGJ_SPH: Self = Self::Symbol("int1e_zz_origj_sph");
    pub const INT1E_ZZ_ORIGJ_SPINOR: Self = Self::Symbol("int1e_zz_origj_spinor");

    // Phase 24 Cluster A high-rank tensors + trace contractions (MOM-02/03).
    // rrr/rrrr have NO `_origj` symbol in libcint 6.1.3 (OQ-3) — not registered.
    pub const INT1E_RR_CART: Self = Self::Symbol("int1e_rr_cart");
    pub const INT1E_RR_SPH: Self = Self::Symbol("int1e_rr_sph");
    pub const INT1E_RR_SPINOR: Self = Self::Symbol("int1e_rr_spinor");

    pub const INT1E_RRR_CART: Self = Self::Symbol("int1e_rrr_cart");
    pub const INT1E_RRR_SPH: Self = Self::Symbol("int1e_rrr_sph");
    pub const INT1E_RRR_SPINOR: Self = Self::Symbol("int1e_rrr_spinor");

    pub const INT1E_RRRR_CART: Self = Self::Symbol("int1e_rrrr_cart");
    pub const INT1E_RRRR_SPH: Self = Self::Symbol("int1e_rrrr_sph");
    pub const INT1E_RRRR_SPINOR: Self = Self::Symbol("int1e_rrrr_spinor");

    pub const INT1E_R2_CART: Self = Self::Symbol("int1e_r2_cart");
    pub const INT1E_R2_SPH: Self = Self::Symbol("int1e_r2_sph");
    pub const INT1E_R2_SPINOR: Self = Self::Symbol("int1e_r2_spinor");

    pub const INT1E_R4_CART: Self = Self::Symbol("int1e_r4_cart");
    pub const INT1E_R4_SPH: Self = Self::Symbol("int1e_r4_sph");
    pub const INT1E_R4_SPINOR: Self = Self::Symbol("int1e_r4_spinor");

    pub const INT1E_RR_ORIGJ_CART: Self = Self::Symbol("int1e_rr_origj_cart");
    pub const INT1E_RR_ORIGJ_SPH: Self = Self::Symbol("int1e_rr_origj_sph");
    pub const INT1E_RR_ORIGJ_SPINOR: Self = Self::Symbol("int1e_rr_origj_spinor");

    pub const INT1E_R2_ORIGJ_CART: Self = Self::Symbol("int1e_r2_origj_cart");
    pub const INT1E_R2_ORIGJ_SPH: Self = Self::Symbol("int1e_r2_origj_sph");
    pub const INT1E_R2_ORIGJ_SPINOR: Self = Self::Symbol("int1e_r2_origj_spinor");

    pub const INT1E_R4_ORIGJ_CART: Self = Self::Symbol("int1e_r4_origj_cart");
    pub const INT1E_R4_ORIGJ_SPH: Self = Self::Symbol("int1e_r4_origj_sph");
    pub const INT1E_R4_ORIGJ_SPINOR: Self = Self::Symbol("int1e_r4_origj_spinor");

    // Phase 23 both-side rank-9 1e families (spinor returns UnsupportedApi, D-06).
    pub const INT1E_IPOVLPIP_CART: Self = Self::Symbol("int1e_ipovlpip_cart");
    pub const INT1E_IPOVLPIP_SPH: Self = Self::Symbol("int1e_ipovlpip_sph");
    pub const INT1E_IPOVLPIP_SPINOR: Self = Self::Symbol("int1e_ipovlpip_spinor");

    pub const INT1E_IPKINIP_CART: Self = Self::Symbol("int1e_ipkinip_cart");
    pub const INT1E_IPKINIP_SPH: Self = Self::Symbol("int1e_ipkinip_sph");
    pub const INT1E_IPKINIP_SPINOR: Self = Self::Symbol("int1e_ipkinip_spinor");

    pub const INT1E_IPNUCIP_CART: Self = Self::Symbol("int1e_ipnucip_cart");
    pub const INT1E_IPNUCIP_SPH: Self = Self::Symbol("int1e_ipnucip_sph");
    pub const INT1E_IPNUCIP_SPINOR: Self = Self::Symbol("int1e_ipnucip_spinor");

    // Phase 25 HESS-01 bra-only rank-9 1e Hessian families (∇²bra, component_rank=9;
    // spinor returns UnsupportedApi, D-11). ovlp/kin ride the no-Rys overlap-deriv
    // engine; nuc/rinv ride the nuclear/Rys 1e path.
    pub const INT1E_IPIPOVLP_CART: Self = Self::Symbol("int1e_ipipovlp_cart");
    pub const INT1E_IPIPOVLP_SPH: Self = Self::Symbol("int1e_ipipovlp_sph");
    pub const INT1E_IPIPOVLP_SPINOR: Self = Self::Symbol("int1e_ipipovlp_spinor");

    pub const INT1E_IPIPNUC_CART: Self = Self::Symbol("int1e_ipipnuc_cart");
    pub const INT1E_IPIPNUC_SPH: Self = Self::Symbol("int1e_ipipnuc_sph");
    pub const INT1E_IPIPNUC_SPINOR: Self = Self::Symbol("int1e_ipipnuc_spinor");

    pub const INT1E_IPIPKIN_CART: Self = Self::Symbol("int1e_ipipkin_cart");
    pub const INT1E_IPIPKIN_SPH: Self = Self::Symbol("int1e_ipipkin_sph");
    pub const INT1E_IPIPKIN_SPINOR: Self = Self::Symbol("int1e_ipipkin_spinor");

    pub const INT1E_IPIPRINV_CART: Self = Self::Symbol("int1e_ipiprinv_cart");
    pub const INT1E_IPIPRINV_SPH: Self = Self::Symbol("int1e_ipiprinv_sph");
    pub const INT1E_IPIPRINV_SPINOR: Self = Self::Symbol("int1e_ipiprinv_spinor");

    // Phase 25 HESS-04 3rd-order (deriv3.c, rank 27): ∇∇∇ on bra/ket per family.
    // ipipipnuc/ipipiprinv = bra ∇∇∇; ipipnucip/ipiprinvip = bra ∇∇ + ket ∇.
    pub const INT1E_IPIPIPNUC_CART: Self = Self::Symbol("int1e_ipipipnuc_cart");
    pub const INT1E_IPIPIPNUC_SPH: Self = Self::Symbol("int1e_ipipipnuc_sph");
    pub const INT1E_IPIPIPNUC_SPINOR: Self = Self::Symbol("int1e_ipipipnuc_spinor");

    pub const INT1E_IPIPIPRINV_CART: Self = Self::Symbol("int1e_ipipiprinv_cart");
    pub const INT1E_IPIPIPRINV_SPH: Self = Self::Symbol("int1e_ipipiprinv_sph");
    pub const INT1E_IPIPIPRINV_SPINOR: Self = Self::Symbol("int1e_ipipiprinv_spinor");

    pub const INT1E_IPIPNUCIP_CART: Self = Self::Symbol("int1e_ipipnucip_cart");
    pub const INT1E_IPIPNUCIP_SPH: Self = Self::Symbol("int1e_ipipnucip_sph");
    pub const INT1E_IPIPNUCIP_SPINOR: Self = Self::Symbol("int1e_ipipnucip_spinor");

    pub const INT1E_IPIPRINVIP_CART: Self = Self::Symbol("int1e_ipiprinvip_cart");
    pub const INT1E_IPIPRINVIP_SPH: Self = Self::Symbol("int1e_ipiprinvip_sph");
    pub const INT1E_IPIPRINVIP_SPINOR: Self = Self::Symbol("int1e_ipiprinvip_spinor");

    // Phase 25 HESS-04 4th-order (deriv4.c, rank 81): bra+2 AND ket+2 dual
    // headroom. ipipipiprinv = bra ∇∇∇∇; ipiprinvipip = ket ∇∇ + bra ∇∇;
    // ipipiprinvip = bra ∇∇∇ + ket ∇.
    pub const INT1E_IPIPIPIPRINV_CART: Self = Self::Symbol("int1e_ipipipiprinv_cart");
    pub const INT1E_IPIPIPIPRINV_SPH: Self = Self::Symbol("int1e_ipipipiprinv_sph");
    pub const INT1E_IPIPIPIPRINV_SPINOR: Self = Self::Symbol("int1e_ipipipiprinv_spinor");

    pub const INT1E_IPIPRINVIPIP_CART: Self = Self::Symbol("int1e_ipiprinvipip_cart");
    pub const INT1E_IPIPRINVIPIP_SPH: Self = Self::Symbol("int1e_ipiprinvipip_sph");
    pub const INT1E_IPIPRINVIPIP_SPINOR: Self = Self::Symbol("int1e_ipiprinvipip_spinor");

    pub const INT1E_IPIPIPRINVIP_CART: Self = Self::Symbol("int1e_ipipiprinvip_cart");
    pub const INT1E_IPIPIPRINVIP_SPH: Self = Self::Symbol("int1e_ipipiprinvip_sph");
    pub const INT1E_IPIPIPRINVIP_SPINOR: Self = Self::Symbol("int1e_ipipiprinvip_spinor");

    pub const INT2E_IP1_CART: Self = Self::Symbol("int2e_ip1_cart");
    pub const INT2E_IP1_SPH: Self = Self::Symbol("int2e_ip1_sph");
    pub const INT2E_IP1_SPINOR: Self = Self::Symbol("int2e_ip1_spinor");

    // Phase 23 DRV1-01: int2e_ip2 (∇ on the ket bra-center k). Spinor registered
    // for surface completeness; kernel returns UnsupportedApi (D-06).
    pub const INT2E_IP2_CART: Self = Self::Symbol("int2e_ip2_cart");
    pub const INT2E_IP2_SPH: Self = Self::Symbol("int2e_ip2_sph");
    pub const INT2E_IP2_SPINOR: Self = Self::Symbol("int2e_ip2_spinor");

    // Phase 25 HESS-02: 2e Hessian set, promoted to STABLE (D-07). int2e_ipip1
    // (∇²bra-i) and int2e_ipvip1 (∇_i∇_j) were sph-only `unstable::source::2e`
    // stubs — re-homed here as one canonical stable entry per symbol (no alias,
    // no unstable gate). int2e_ip1ip2 (∇_i∇_k, rank 9) and int2e_ipip1ipip2
    // (∇²_i∇²_k, rank 81, 4th-order 2e) are registered fresh. All host-routed
    // through fill_g_tensor_2e (FND-02). Spinor → UnsupportedApi (D-11).
    pub const INT2E_IPIP1_CART: Self = Self::Symbol("int2e_ipip1_cart");
    pub const INT2E_IPIP1_SPH: Self = Self::Symbol("int2e_ipip1_sph");
    pub const INT2E_IPIP1_SPINOR: Self = Self::Symbol("int2e_ipip1_spinor");

    pub const INT2E_IPVIP1_CART: Self = Self::Symbol("int2e_ipvip1_cart");
    pub const INT2E_IPVIP1_SPH: Self = Self::Symbol("int2e_ipvip1_sph");
    pub const INT2E_IPVIP1_SPINOR: Self = Self::Symbol("int2e_ipvip1_spinor");

    pub const INT2E_IP1IP2_CART: Self = Self::Symbol("int2e_ip1ip2_cart");
    pub const INT2E_IP1IP2_SPH: Self = Self::Symbol("int2e_ip1ip2_sph");
    pub const INT2E_IP1IP2_SPINOR: Self = Self::Symbol("int2e_ip1ip2_spinor");

    pub const INT2E_IPIP1IPIP2_CART: Self = Self::Symbol("int2e_ipip1ipip2_cart");
    pub const INT2E_IPIP1IPIP2_SPH: Self = Self::Symbol("int2e_ipip1ipip2_sph");
    pub const INT2E_IPIP1IPIP2_SPINOR: Self = Self::Symbol("int2e_ipip1ipip2_spinor");

    // Phase 23 DRV1-04: int2c2e_ip1 (∇ on bra center i) + int2c2e_ip2 (∇ on ket
    // center k). Spinor registered for surface completeness (D-06).
    pub const INT2C2E_IP1_CART: Self = Self::Symbol("int2c2e_ip1_cart");
    pub const INT2C2E_IP1_SPH: Self = Self::Symbol("int2c2e_ip1_sph");
    pub const INT2C2E_IP1_SPINOR: Self = Self::Symbol("int2c2e_ip1_spinor");

    pub const INT2C2E_IP2_CART: Self = Self::Symbol("int2c2e_ip2_cart");
    pub const INT2C2E_IP2_SPH: Self = Self::Symbol("int2c2e_ip2_sph");
    pub const INT2C2E_IP2_SPINOR: Self = Self::Symbol("int2c2e_ip2_spinor");

    // Phase 23 DRV1-05: int3c2e_ip2 (∇ on the auxiliary `k` center). cintx maps the
    // real aux k into the 2e `ll` slot, so the derivative is applied via `nabla1l_2e`
    // (RESEARCH Pitfall 2). Spinor registered for surface completeness; kernel
    // returns UnsupportedApi (D-06).
    pub const INT3C2E_IP2_CART: Self = Self::Symbol("int3c2e_ip2_cart");
    pub const INT3C2E_IP2_SPH: Self = Self::Symbol("int3c2e_ip2_sph");
    pub const INT3C2E_IP2_SPINOR: Self = Self::Symbol("int3c2e_ip2_spinor");

    // Phase 23 DRV1-03: int3c1e_ip1 (∇ on bra i of the 3-center OVERLAP, no Rys)
    // and int3c1e_iprinv (∇ on bra i of the 3-center rinv-COULOMB, Rys-driven via
    // the existing PTR_RINV_ORIG env slot, D-08). Both arity-3, rank-3. Spinor
    // registered for surface completeness; kernel returns UnsupportedApi (D-06).
    pub const INT3C1E_IP1_CART: Self = Self::Symbol("int3c1e_ip1_cart");
    pub const INT3C1E_IP1_SPH: Self = Self::Symbol("int3c1e_ip1_sph");
    pub const INT3C1E_IP1_SPINOR: Self = Self::Symbol("int3c1e_ip1_spinor");

    pub const INT3C1E_IPRINV_CART: Self = Self::Symbol("int3c1e_iprinv_cart");
    pub const INT3C1E_IPRINV_SPH: Self = Self::Symbol("int3c1e_iprinv_sph");
    pub const INT3C1E_IPRINV_SPINOR: Self = Self::Symbol("int3c1e_iprinv_spinor");

    pub const INT1E_ECP_IPRINV_CART: Self = Self::Symbol("int1e_ecp_iprinv_cart");
    pub const INT1E_ECP_IPRINV_SPH: Self = Self::Symbol("int1e_ecp_iprinv_sph");
    /// Spinor form registered for surface completeness; kernel returns UnsupportedApi (D-03/R5).
    pub const INT1E_ECP_IPRINV_SPINOR: Self = Self::Symbol("int1e_ecp_iprinv_spinor");

    // Phase 25 HESS-03: multi-center rank-9 Hessian families (component_rank=9).
    // int2c2e_ipip1 / int3c2e_ipip1 raise the bra center-1 headroom (i_inc=2);
    // int3c2e_ipip2 raises the KET (auxiliary k → 2e `ll` slot) headroom
    // (k_inc=2, third ng[] element — D-09 ket-side). Spinor reps registered for
    // surface completeness; kernel returns UnsupportedApi (D-11).
    pub const INT2C2E_IPIP1_CART: Self = Self::Symbol("int2c2e_ipip1_cart");
    pub const INT2C2E_IPIP1_SPH: Self = Self::Symbol("int2c2e_ipip1_sph");

    pub const INT3C2E_IPIP1_CART: Self = Self::Symbol("int3c2e_ipip1_cart");
    pub const INT3C2E_IPIP1_SPH: Self = Self::Symbol("int3c2e_ipip1_sph");

    pub const INT3C2E_IPIP2_CART: Self = Self::Symbol("int3c2e_ipip2_cart");
    pub const INT3C2E_IPIP2_SPH: Self = Self::Symbol("int3c2e_ipip2_sph");

    fn symbol(self) -> &'static str {
        match self {
            Self::Symbol(symbol) => symbol,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawEvalSummary {
    pub not0: i32,
    pub bytes_written: usize,
    pub workspace_bytes: usize,
}

struct ResolvedRawApi {
    descriptor: &'static OperatorDescriptor,
    representation: Representation,
}

struct PreparedRawCall {
    op: OperatorId,
    representation: Representation,
    basis: BasisSet,
    shells: ShellTuple,
    query: WorkspaceQuery,
    compat_dims: CompatDims,
    _options: ExecutionOptions,
}

/// Raw atom view over libcint-style `atm` slots.
#[derive(Clone, Copy, Debug)]
pub struct RawAtmView<'a> {
    data: &'a [i32],
}

impl<'a> RawAtmView<'a> {
    pub fn new(data: &'a [i32]) -> Result<Self, cintxRsError> {
        if data.len() % ATM_SLOTS != 0 {
            return Err(cintxRsError::InvalidAtmLayout {
                slot_width: ATM_SLOTS,
                provided: data.len(),
            });
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len() / ATM_SLOTS
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<RawAtmRecord<'a>> {
        let start = index.checked_mul(ATM_SLOTS)?;
        let record = self.data.get(start..start + ATM_SLOTS)?;
        Some(RawAtmRecord { record })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RawAtmRecord<'a>> {
        self.data
            .chunks_exact(ATM_SLOTS)
            .map(|record| RawAtmRecord { record })
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        for record in self.iter() {
            record.validate(env)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawAtmRecord<'a> {
    record: &'a [i32],
}

impl<'a> RawAtmRecord<'a> {
    pub fn charge(&self) -> i32 {
        self.record[CHARGE_OF]
    }

    pub fn coord_offset(&self) -> i32 {
        self.record[PTR_COORD]
    }

    pub fn nuclear_model_raw(&self) -> i32 {
        self.record[NUC_MOD_OF]
    }

    pub fn zeta_offset(&self) -> i32 {
        self.record[PTR_ZETA]
    }

    pub fn fractional_charge_offset(&self) -> i32 {
        self.record[PTR_FRAC_CHARGE]
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        env.validate_range("PTR_COORD", self.coord_offset(), 3)?;
        match self.nuclear_model_raw() {
            POINT_NUC => {}
            GAUSSIAN_NUC => {
                env.validate_scalar("PTR_ZETA", self.zeta_offset())?;
            }
            FRAC_CHARGE_NUC => {
                env.validate_scalar("PTR_FRAC_CHARGE", self.fractional_charge_offset())?;
            }
            other => {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("unsupported nuclear model {other}"),
                });
            }
        }
        Ok(())
    }
}

/// Raw basis-shell view over libcint-style `bas` slots.
#[derive(Clone, Copy, Debug)]
pub struct RawBasView<'a> {
    data: &'a [i32],
}

impl<'a> RawBasView<'a> {
    pub fn new(data: &'a [i32]) -> Result<Self, cintxRsError> {
        if data.len() % BAS_SLOTS != 0 {
            return Err(cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: data.len(),
            });
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len() / BAS_SLOTS
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<RawBasRecord<'a>> {
        let start = index.checked_mul(BAS_SLOTS)?;
        let record = self.data.get(start..start + BAS_SLOTS)?;
        Some(RawBasRecord { record })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RawBasRecord<'a>> {
        self.data
            .chunks_exact(BAS_SLOTS)
            .map(|record| RawBasRecord { record })
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        for record in self.iter() {
            record.validate(env)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawBasRecord<'a> {
    record: &'a [i32],
}

impl<'a> RawBasRecord<'a> {
    pub fn atom_index_raw(&self) -> i32 {
        self.record[ATOM_OF]
    }

    pub fn ang_momentum_raw(&self) -> i32 {
        self.record[ANG_OF]
    }

    pub fn nprim_raw(&self) -> i32 {
        self.record[NPRIM_OF]
    }

    pub fn nctr_raw(&self) -> i32 {
        self.record[NCTR_OF]
    }

    pub fn kappa_raw(&self) -> i32 {
        self.record[KAPPA_OF]
    }

    pub fn exp_offset(&self) -> i32 {
        self.record[PTR_EXP]
    }

    pub fn coeff_offset(&self) -> i32 {
        self.record[PTR_COEFF]
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        let nprim =
            usize::try_from(self.nprim_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: self.nprim_raw().unsigned_abs() as usize,
            })?;
        let nctr =
            usize::try_from(self.nctr_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: self.nctr_raw().unsigned_abs() as usize,
            })?;

        if nprim == 0 || nctr == 0 {
            return Err(cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: 0,
            });
        }

        env.validate_range("PTR_EXP", self.exp_offset(), nprim)?;
        let coeff_len = nprim
            .checked_mul(nctr)
            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                from: "raw_bas",
                detail: "coefficient range overflowed usize".to_owned(),
            })?;
        env.validate_range("PTR_COEFF", self.coeff_offset(), coeff_len)?;
        Ok(())
    }
}

/// Typed view over a flat libcint-style `ecpbas` slab.
///
/// ECP basis rows reuse the existing `BAS_SLOTS = 8` row width (slots 3 and
/// 4 are reinterpreted as `RADI_POWER` and `SO_TYPE_OF`). The PySCF ECP
/// kernel reads the slab pointed to by `env[AS_ECPBAS_OFFSET]` with
/// `env[AS_NECPBAS]` rows; this view is the typed surface for safe-Rust
/// consumers.
#[derive(Clone, Copy, Debug)]
pub struct EcpBasArray<'a> {
    data: &'a [i32],
}

impl<'a> EcpBasArray<'a> {
    /// Construct an `EcpBasArray` from a flat `i32` slab. Returns
    /// `cintxRsError::InvalidBasLayout` if `data.len()` is not a multiple
    /// of `BAS_SLOTS` (same error variant `RawBasView::new` uses).
    pub fn new(data: &'a [i32]) -> Result<Self, cintxRsError> {
        if data.len() % BAS_SLOTS != 0 {
            return Err(cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: data.len(),
            });
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len() / BAS_SLOTS
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the raw 8-slot row at `index`, or panics if out of bounds —
    /// callers must check `len()` first. Mirrors the indexing contract of
    /// `RawBasView::get`.
    pub fn row(&self, index: usize) -> &[i32] {
        let start = index
            .checked_mul(BAS_SLOTS)
            .expect("ecpbas row index overflow");
        &self.data[start..start + BAS_SLOTS]
    }

    /// Reads the `RADI_POWER` slot for row `index`.
    pub fn radial_power(&self, index: usize) -> i32 {
        self.row(index)[RADI_POWER]
    }

    /// Reads the `SO_TYPE_OF` slot for row `index`.
    pub fn so_type(&self, index: usize) -> i32 {
        self.row(index)[SO_TYPE_OF]
    }

    /// Iterates over the raw 8-slot rows.
    pub fn iter_rows(&self) -> std::slice::ChunksExact<'_, i32> {
        self.data.chunks_exact(BAS_SLOTS)
    }
}

/// Raw environment view over libcint-style `env` values.
#[derive(Clone, Copy, Debug)]
pub struct RawEnvView<'a> {
    data: &'a [f64],
}

impl<'a> RawEnvView<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    pub fn validate_scalar(&self, slot: &'static str, offset: i32) -> Result<usize, cintxRsError> {
        self.validate_range(slot, offset, 1)
    }

    pub fn validate_range(
        &self,
        slot: &'static str,
        offset: i32,
        len: usize,
    ) -> Result<usize, cintxRsError> {
        let start = normalize_offset(slot, offset, self.len())?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| cintxRsError::InvalidEnvOffset {
                slot,
                offset: start,
                env_len: self.len(),
            })?;
        if end > self.len() {
            return Err(cintxRsError::InvalidEnvOffset {
                slot,
                offset: start,
                env_len: self.len(),
            });
        }
        Ok(start)
    }

    pub fn slice(
        &self,
        slot: &'static str,
        offset: i32,
        len: usize,
    ) -> Result<&'a [f64], cintxRsError> {
        let start = self.validate_range(slot, offset, len)?;
        Ok(&self.data[start..start + len])
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn query_workspace_raw(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<WorkspaceQuery, cintxRsError> {
    let prepared = prepare_raw_call(api, dims, shls, atm, bas, env, opt)?;
    Ok(prepared.query)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn eval_raw(
    api: RawApiId,
    out: Option<&mut [f64]>,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
    cache: Option<&mut [f64]>,
) -> Result<RawEvalSummary, cintxRsError> {
    let prepared = prepare_raw_call(api, dims, shls, atm, bas, env, opt)?;

    if let Some(out_buffer) = out.as_ref() {
        prepared.compat_dims.ensure_output_len(out_buffer.len())?;
    } else {
        return Ok(RawEvalSummary {
            not0: 0,
            bytes_written: 0,
            workspace_bytes: prepared.query.bytes,
        });
    }

    if let Some(cache) = cache {
        ensure_cache_len(prepared.query.bytes, cache.len())?;
    }

    let mut plan = ExecutionPlan::new(
        prepared.op,
        prepared.representation,
        &prepared.basis,
        prepared.shells.clone(),
        &prepared.query,
    )?;

    // Extract f12_zeta from env[PTR_F12_ZETA] for F12/STG/YP integrals (raw compat path).
    // Raw callers are expected to set env[9] = zeta before calling any F12 integral.
    // The manifest canonical_family for STG/YP operators is "f12"; operator_name is "stg"/"yp".
    // We detect F12 symbols by their full symbol name prefix (int2e_stg / int2e_yp).
    if is_f12_family_symbol(plan.descriptor.operator_symbol()) {
        let zeta = env.get(PTR_F12_ZETA).copied().unwrap_or(0.0);
        plan.operator_env_params.f12_zeta = Some(zeta);
        // Validate before dispatch so we return a typed error on bad input.
        cintx_runtime::validator::validate_f12_env_params("f12", &plan.operator_env_params)?;
    }
    // Phase 21-01: Extract rinv_orig from env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] for iprinv operators.
    // Raw callers must set env[4..7] = [x, y, z] (in Bohr) before calling any iprinv integral.
    // Guard with env.len() >= PTR_RINV_ORIG + 3 so a too-short env never indexes out of bounds
    // (T-21-01-01); if the origin is still None after the read, validate_rinv_orig_env_params
    // returns a typed InvalidEnvParam BEFORE kernel entry — no garbage-origin evaluation (T-21-01-02).
    // Phase 24 D-04 / OQ-1: plain int1e_rinv / int1e_drinv are single-center 1/r
    // potentials. They read the SAME rinv-origin slot (env[PTR_RINV_ORIG], i.e.
    // env[4..6]) as iprinv — NOT the gauge/common origin slot (env[1..3]) used by
    // the moment families. The rinv center is read here so the kernel evaluates at
    // a caller-supplied non-zero origin; a zero origin is trivially-passing and the
    // parity tests inject a non-zero center via env_with_rinv_origin.
    let sym = plan.descriptor.operator_symbol();
    if is_iprinv_family_symbol(sym) || is_rinv_family_symbol(sym) {
        if env.len() >= PTR_RINV_ORIG + 3 {
            let x = env[PTR_RINV_ORIG];
            let y = env[PTR_RINV_ORIG + 1];
            let z = env[PTR_RINV_ORIG + 2];
            plan.operator_env_params.rinv_orig = Some([x, y, z]);
        }
        cintx_runtime::validator::validate_rinv_orig_env_params(
            plan.descriptor.operator_name(),
            &plan.operator_env_params,
        )?;
    }
    // Phase 22 FND-01: extract common_orig (gauge origin) from env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3].
    // D-02: operator-AGNOSTIC — read unconditionally; no operator-name guard exists yet (moments/GIAO
    // add their own dispatch in Phases 24/26). Only the bounds guard (T-22-01) prevents OOB indexing.
    if env.len() >= PTR_COMMON_ORIG + 3 {
        let x = env[PTR_COMMON_ORIG];
        let y = env[PTR_COMMON_ORIG + 1];
        let z = env[PTR_COMMON_ORIG + 2];
        plan.operator_env_params.common_orig = Some([x, y, z]);
    }
    cintx_runtime::validator::validate_common_orig_env_params(
        plan.descriptor.operator_name(),
        &plan.operator_env_params,
    )?;
    // Phase 19 D-05: ECP dispatch guard — reject before kernel launch when
    // env[AS_NECPBAS] is missing/zero/non-finite. Mirrors the F12 zeta gate
    // above (same insertion point, same error variant). Plan 04 wires the
    // kernel-side reader that consumes env[AS_ECPBAS_OFFSET].
    if is_ecp_family_symbol(plan.descriptor.operator_symbol()) {
        let necpbas = env.get(AS_NECPBAS).copied().unwrap_or(0.0);
        if !necpbas.is_finite() || necpbas <= 0.0 {
            return Err(cintxRsError::InvalidEnvParam {
                param: "AS_NECPBAS",
                reason: format!(
                    "env[AS_NECPBAS] must be > 0 and finite for ECP operators, got {necpbas}"
                ),
            });
        }
    }
    if plan.descriptor.entry.canonical_family == "grids" {
        plan.operator_env_params.grids_params = Some(extract_grids_env_params(env)?);
        cintx_runtime::validator::validate_grids_env_params("grids", &plan.operator_env_params)?;
    }

    let executor = CubeClExecutor::new();
    let mut allocator = HostWorkspaceAllocator::default();

    // Allocate the full staging accumulator that we own, so we can read values after execute().
    // RecordingExecutor is not needed: we construct ExecutionIo with our own staging slice and
    // read it directly after executor.execute() returns for each chunk.
    //
    // The grids family emits one AO block per grid point (an extra leading
    // NGRIDS axis from env that the planner's shell-tuple sizing does not see),
    // so scale the accumulator by NGRIDS to match the grids kernel's
    // `ncomp * ngrids * ni * nj` output. `grids_params` was populated above; any
    // other family keeps a factor of 1.
    let grids_repeat = plan
        .operator_env_params
        .grids_params
        .as_ref()
        .map(|params| params.ngrids.max(1))
        .unwrap_or(1);
    let staging_elements = plan
        .output_layout
        .staging_elements
        .checked_mul(grids_repeat)
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "compat_raw",
            detail: "grids staging element count overflowed usize".to_owned(),
        })?;
    let mut staging = Vec::new();
    staging.try_reserve_exact(staging_elements).map_err(|_| {
        cintxRsError::HostAllocationFailed {
            bytes: staging_elements.saturating_mul(size_of::<f64>()),
        }
    })?;
    staging.resize(staging_elements, 0.0);

    if !executor.supports(&plan) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "{}/{}/{}",
                plan.descriptor.family(),
                plan.descriptor.operator_name(),
                plan.representation
            ),
        });
    }

    let backend_workspace = executor.query_workspace(&plan)?.get();
    if backend_workspace > plan.workspace.bytes {
        return Err(cintxRsError::MemoryLimitExceeded {
            requested: backend_workspace,
            limit: plan.workspace.bytes,
        });
    }

    let schedule = schedule_chunks(&plan.workspace);
    let total_units = plan.workspace.work_units.max(1);

    let mut total_not0: i32 = 0;
    let mut total_transfer_bytes: usize = 0;

    for chunk in schedule.chunks() {
        // Compute staging slice range for this chunk (mirrors staging_elements_for_chunk logic).
        let start = chunk.work_unit_start.min(total_units);
        let end = chunk
            .work_unit_start
            .saturating_add(chunk.work_unit_count)
            .min(total_units);
        let prefix = staging_elements.saturating_mul(start) / total_units;
        let suffix = staging_elements.saturating_mul(end) / total_units;
        let chunk_len = suffix.saturating_sub(prefix).max(1);

        // Allocate the chunk staging slice and workspace.
        let chunk_staging_bytes = chunk_len
            .checked_mul(size_of::<f64>())
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
        let mut chunk_staging = Vec::new();
        chunk_staging.try_reserve_exact(chunk_len).map_err(|_| {
            cintxRsError::HostAllocationFailed {
                bytes: chunk_staging_bytes,
            }
        })?;
        chunk_staging.resize(chunk_len, 0.0);

        let mut workspace = allocator.try_alloc(chunk.bytes, plan.workspace.alignment)?;

        {
            let mut io =
                ExecutionIo::new(chunk, &mut chunk_staging, &mut workspace, plan.dispatch)?;
            io.ensure_output_contract()?;
            let chunk_stats = executor.execute(&plan, &mut io)?;
            total_not0 = total_not0.saturating_add(chunk_stats.not0.max(0));
            total_transfer_bytes = total_transfer_bytes.saturating_add(io.transfer_bytes());
        }
        allocator.release(workspace);

        // Copy chunk staging into the appropriate range of the accumulator.
        let dest_end = prefix.saturating_add(chunk_len).min(staging_elements);
        if prefix < dest_end {
            staging[prefix..dest_end].copy_from_slice(&chunk_staging[..dest_end - prefix]);
        }
    }

    let out = out.expect("checked out.is_some()");
    let written_elements = prepared.compat_dims.write(out, &staging)?;
    let bytes_written = written_elements
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "compat_raw",
            detail: "written byte count overflowed usize".to_owned(),
        })?;

    Ok(RawEvalSummary {
        not0: total_not0,
        bytes_written,
        workspace_bytes: plan.workspace.bytes,
    })
}

fn active_manifest_profile() -> &'static str {
    match (cfg!(feature = "with-f12"), cfg!(feature = "with-4c1e")) {
        (true, true) => "with-f12+with-4c1e",
        (true, false) => "with-f12",
        (false, true) => "with-4c1e",
        (false, false) => "base",
    }
}

fn unstable_source_api_enabled() -> bool {
    cfg!(feature = "unstable-source-api")
}

fn is_f12_family_symbol(symbol: &str) -> bool {
    symbol.starts_with("int2e_stg") || symbol.starts_with("int2e_yp")
}

/// Phase 19: identifies the four ECP operator symbols
/// (`int1e_ecp_cart`, `int1e_ecp_sph`, `int1e_ecp_ipnuc_cart`,
/// `int1e_ecp_ipnuc_sph`). Mirrors `is_f12_family_symbol` — sibling
/// gating insertion point in `eval_raw`.
fn is_ecp_family_symbol(symbol: &str) -> bool {
    symbol.starts_with("int1e_ecp_")
}

/// Phase 21-01: identifies iprinv-family operator symbols.
///
/// Returns `true` for any symbol whose name contains `"iprinv"` — covers
/// `int1e_iprinv_{cart,sph,spinor}` and `int1e_ecp_iprinv_{cart,sph}`.
/// Used to gate the PTR_RINV_ORIG env-read block in `eval_raw`.
fn is_iprinv_family_symbol(symbol: &str) -> bool {
    symbol.contains("iprinv")
}

/// Phase 24 D-04 / OQ-1: identifies plain rinv-family operator symbols
/// (`int1e_rinv_*`, `int1e_drinv_*`) that read PTR_RINV_ORIG (env[4..6]).
///
/// Distinct from [`is_iprinv_family_symbol`]: matches only the plain single-center
/// `int1e_rinv` / `int1e_drinv` families, NOT the gradient `iprinv` family (which
/// has its own gate). `int1e_drinv` contains the substring `rinv` but is matched
/// explicitly by prefix so the two gates never overlap.
fn is_rinv_family_symbol(symbol: &str) -> bool {
    symbol.starts_with("int1e_rinv_") || symbol.starts_with("int1e_drinv_")
}

fn parse_env_usize_param(
    env: &[f64],
    index: usize,
    param: &'static str,
) -> Result<usize, cintxRsError> {
    let value = *env
        .get(index)
        .ok_or_else(|| cintxRsError::InvalidEnvParam {
            param,
            reason: format!("env[{index}] is missing"),
        })?;
    if !value.is_finite() {
        return Err(cintxRsError::InvalidEnvParam {
            param,
            reason: format!("env[{index}] must be finite, got {value}"),
        });
    }
    if value < 0.0 {
        return Err(cintxRsError::InvalidEnvParam {
            param,
            reason: format!("env[{index}] must be >= 0, got {value}"),
        });
    }
    if value.fract() != 0.0 {
        return Err(cintxRsError::InvalidEnvParam {
            param,
            reason: format!("env[{index}] must be an integer, got {value}"),
        });
    }
    if value > (usize::MAX as f64) {
        return Err(cintxRsError::InvalidEnvParam {
            param,
            reason: format!("env[{index}] exceeds usize::MAX: {value}"),
        });
    }
    Ok(value as usize)
}

fn extract_grids_env_params(env: &[f64]) -> Result<GridsEnvParams, cintxRsError> {
    let ngrids = parse_env_usize_param(env, NGRIDS, "NGRIDS")?;
    if ngrids == 0 {
        return Err(cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            reason: "NGRIDS must be > 0 for grids integrals".to_owned(),
        });
    }
    let ptr_grids = parse_env_usize_param(env, PTR_GRIDS, "PTR_GRIDS")?;
    let coord_len = ngrids
        .checked_mul(3)
        .ok_or_else(|| cintxRsError::InvalidEnvParam {
            param: "PTR_GRIDS",
            reason: format!("NGRIDS={ngrids} overflows when expanded to xyz coordinates"),
        })?;
    let coord_end =
        ptr_grids
            .checked_add(coord_len)
            .ok_or_else(|| cintxRsError::InvalidEnvParam {
                param: "PTR_GRIDS",
                reason: format!("PTR_GRIDS={ptr_grids} + 3*NGRIDS={coord_len} overflowed"),
            })?;
    if coord_end > env.len() {
        return Err(cintxRsError::InvalidEnvParam {
            param: "PTR_GRIDS",
            reason: format!(
                "grid coordinate range [{ptr_grids}..{coord_end}) exceeds env length {}",
                env.len()
            ),
        });
    }

    let mut grid_coords = Vec::new();
    grid_coords
        .try_reserve_exact(ngrids)
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: ngrids.saturating_mul(size_of::<[f64; 3]>()),
        })?;
    for (index, chunk) in env[ptr_grids..coord_end].chunks_exact(3).enumerate() {
        if !chunk.iter().all(|value| value.is_finite()) {
            return Err(cintxRsError::InvalidEnvParam {
                param: "PTR_GRIDS",
                reason: format!(
                    "grid coordinate {index} contains non-finite values: [{}, {}, {}]",
                    chunk[0], chunk[1], chunk[2]
                ),
            });
        }
        grid_coords.push([chunk[0], chunk[1], chunk[2]]);
    }

    Ok(GridsEnvParams {
        ngrids,
        ptr_grids,
        grid_coords,
    })
}

/// Output replication factor contributed by the grids `NGRIDS` axis.
///
/// The grids family (`int1e_grids*`) emits one AO block per grid point, so the
/// caller-visible output carries an extra leading axis of size `NGRIDS` that
/// lives in `env`, not in the shell tuple or the manifest component rank. The
/// planner sizes the output from the shell tuple + component rank only, so the
/// raw compat layer folds this factor into both the compat component count
/// (output-length contract) and the staging accumulator size.
///
/// Returns `1` for every non-grids family. For the grids family the factor is
/// `NGRIDS`, surfaced through `extract_grids_env_params` so a malformed
/// `NGRIDS`/`PTR_GRIDS` is reported as a typed `InvalidEnvParam` before any
/// allocation or kernel launch.
fn grids_output_repeat(
    descriptor: &OperatorDescriptor,
    env: &[f64],
) -> Result<usize, cintxRsError> {
    if descriptor.entry.canonical_family == "grids" {
        Ok(extract_grids_env_params(env)?.ngrids.max(1))
    } else {
        Ok(1)
    }
}

fn f12_sph_envelope_error(symbol: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("{symbol} is outside with-f12 sph envelope"),
    }
}

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("{VALIDATED_4C1E_REASON} ({reason})"),
    }
}

fn validate_profile_and_source_gate(descriptor: &OperatorDescriptor) -> Result<(), cintxRsError> {
    let symbol = descriptor.operator_symbol();
    let profile = active_manifest_profile();

    // Source-only symbols are gated by the `unstable-source-api` feature. Once that
    // gate passes the symbol must still be compiled into a profile available in THIS
    // build — either the active base/with-* profile (a few source-only `2e` symbols
    // ship there) OR the dedicated `unstable-source` profile (origi/grids/breit/origk/
    // ssc). A source-only symbol compiled into NEITHER is rejected, not silently
    // accepted. CR-01: this membership check was previously unreachable dead code
    // behind an early `return Ok(())`. `active_manifest_profile()` never returns
    // "unstable-source", so checking only that profile (the old dead block's logic)
    // would wrongly reject the base-profile source `2e` symbols — hence the OR.
    if descriptor.is_source_only() {
        if !unstable_source_api_enabled() {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "source-only symbol {symbol} requires feature `unstable-source-api`"
                ),
            });
        }
        if !descriptor.is_compiled_in_profile(profile)
            && !descriptor.is_compiled_in_profile("unstable-source")
        {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "raw api {symbol} is not compiled in active profile {profile} or the unstable-source profile"
                ),
            });
        }
        return Ok(());
    }

    // Non-source-only symbols: check the active compiled profile.
    if !descriptor.is_compiled_in_profile(profile) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("raw api {symbol} is not compiled in active profile {profile}"),
        });
    }

    Ok(())
}

fn dims_match_natural(dims: Option<&[i32]>, natural_extents: &[usize]) -> bool {
    let Some(dims) = dims else {
        return true;
    };
    if dims.len() != natural_extents.len() {
        return false;
    }
    dims.iter()
        .zip(natural_extents.iter())
        .all(|(provided, expected)| usize::try_from(*provided).ok() == Some(*expected))
}

fn validate_f12_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    let symbol = descriptor.operator_symbol();
    if !is_f12_family_symbol(symbol) {
        return Ok(());
    }

    if !matches!(representation, Representation::Spheric) {
        return Err(f12_sph_envelope_error(symbol));
    }
    if !dims_match_natural(dims, natural_extents) {
        return Err(f12_sph_envelope_error(symbol));
    }
    Ok(())
}

fn validate_4c1e_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shells: &ShellTuple,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    if descriptor.entry.canonical_family != "4c1e" {
        return Ok(());
    }

    // D-05: Spinor rejection FIRST — before feature gate check.
    // A Spinor 4c1e request must return UnsupportedApi with "spinor" in the message
    // regardless of whether the with-4c1e feature is enabled.
    if matches!(representation, Representation::Spinor) {
        return Err(validated_4c1e_error(
            "spinor representation not supported for 4c1e",
        ));
    }

    if !cfg!(feature = "with-4c1e") {
        return Err(validated_4c1e_error("with-4c1e feature disabled"));
    }
    if !matches!(
        representation,
        Representation::Cart | Representation::Spheric
    ) {
        return Err(validated_4c1e_error("representation must be cart/sph"));
    }
    if !descriptor.entry.component_rank.trim().is_empty()
        && descriptor.entry.component_rank != "scalar"
    {
        return Err(validated_4c1e_error("component rank must be scalar"));
    }
    if !dims_match_natural(dims, natural_extents) {
        return Err(validated_4c1e_error("dims must be natural"));
    }
    // Validated4C1E requires max(l)<=4.
    if shells.iter().any(|shell| shell.ang_momentum > 4) {
        return Err(validated_4c1e_error("max(l)>4"));
    }
    if CUBECL_RUNTIME_PROFILE != "cpu" {
        return Err(validated_4c1e_error("CubeCL backend must be cpu"));
    }

    Ok(())
}

/// Apply the same manifest profile/source-only/optional envelope policy gates used by
/// compat raw dispatch so safe facade callers get identical UnsupportedApi reasons.
pub fn enforce_safe_facade_policy_gate(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shells: &ShellTuple,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    validate_profile_and_source_gate(descriptor)?;
    validate_f12_envelope(descriptor, representation, None, natural_extents)?;
    validate_4c1e_envelope(descriptor, representation, shells, None, natural_extents)?;
    Ok(())
}

fn prepare_raw_call(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<PreparedRawCall, cintxRsError> {
    let resolved = resolve_raw_api(api)?;
    let atm = RawAtmView::new(atm)?;
    let bas = RawBasView::new(bas)?;
    // Keep the raw env slice for grids NGRIDS extraction before wrapping it in the
    // bounds-checked view (the view does not expose the flat slice).
    let env_slice = env;
    let env = RawEnvView::new(env);

    atm.validate(&env)?;
    bas.validate(&env)?;

    let (basis, shells) = build_typed_basis_and_shell_tuple(
        resolved.descriptor,
        resolved.representation,
        shls,
        &atm,
        &bas,
        &env,
    )?;

    let options = execution_options_from_opt(opt);
    let query = query_workspace(
        resolved.descriptor.id,
        resolved.representation,
        &basis,
        shells.clone(),
        &options,
    )?;

    let layout_plan = ExecutionPlan::new(
        resolved.descriptor.id,
        resolved.representation,
        &basis,
        shells.clone(),
        &query,
    )?;

    validate_f12_envelope(
        resolved.descriptor,
        resolved.representation,
        dims,
        &layout_plan.output_layout.extents,
    )?;
    validate_4c1e_envelope(
        resolved.descriptor,
        resolved.representation,
        &shells,
        dims,
        &layout_plan.output_layout.extents,
    )?;

    // Fold the grids NGRIDS axis into the compat component count so the output
    // length contract covers all ngrids*ncomp*ni*nj elements. libcint lays grids
    // out comp-slowest, then grid, then AO block, and `CompatDims::write` is an
    // order-preserving copy, so this straight fold is byte-exact. Non-grids
    // families get a factor of 1 (component count unchanged).
    let grids_repeat = grids_output_repeat(resolved.descriptor, env_slice)?;
    let component_count = layout_plan
        .component_count
        .checked_mul(grids_repeat)
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "compat_raw",
            detail: "grids component count overflowed usize".to_owned(),
        })?;
    let compat_dims = CompatDims::from_override(
        &layout_plan.output_layout.extents,
        dims,
        component_count,
        layout_plan.output_layout.complex_interleaved,
    )?;

    Ok(PreparedRawCall {
        op: resolved.descriptor.id,
        representation: resolved.representation,
        basis,
        shells,
        query,
        compat_dims,
        _options: options,
    })
}

fn resolve_raw_api(api: RawApiId) -> Result<ResolvedRawApi, cintxRsError> {
    let symbol = api.symbol();
    if is_f12_family_symbol(symbol) && !symbol.ends_with("_sph") {
        return Err(f12_sph_envelope_error(symbol));
    }

    let descriptor =
        Resolver::descriptor_by_symbol(symbol).map_err(|err| map_resolver_error(api, err))?;

    if !matches!(
        descriptor.entry.helper_kind,
        HelperKind::Operator | HelperKind::Legacy | HelperKind::SourceOnly
    ) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "raw api {} must resolve to operator/legacy/source manifest entries",
                symbol
            ),
        });
    }

    validate_profile_and_source_gate(descriptor)?;

    let representation = representation_from_descriptor(descriptor)?;
    Ok(ResolvedRawApi {
        descriptor,
        representation,
    })
}

fn representation_from_descriptor(
    descriptor: &OperatorDescriptor,
) -> Result<Representation, cintxRsError> {
    let rep = descriptor.entry.representation;
    match (rep.cart, rep.spheric, rep.spinor) {
        (true, false, false) => Ok(Representation::Cart),
        (false, true, false) => Ok(Representation::Spheric),
        (false, false, true) => Ok(Representation::Spinor),
        _ => Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "descriptor {} does not map to a single representation",
                descriptor.operator_symbol()
            ),
        }),
    }
}

fn execution_options_from_opt(opt: Option<&RawOptimizerHandle>) -> ExecutionOptions {
    let mut options = ExecutionOptions::default();
    options.profile_label = Some(active_manifest_profile());
    if let Some(opt) = opt {
        options.memory_limit_bytes = opt.workspace_hint_bytes();
    }
    options
}

fn build_typed_basis_and_shell_tuple(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shls: &[i32],
    atm: &RawAtmView<'_>,
    bas: &RawBasView<'_>,
    env: &RawEnvView<'_>,
) -> Result<(BasisSet, ShellTuple), cintxRsError> {
    let mut atoms = Vec::new();
    atoms
        .try_reserve_exact(atm.len())
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: atm.len().saturating_mul(size_of::<Atom>()),
        })?;

    for record in atm.iter() {
        let atomic_number =
            u16::try_from(record.charge()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_atoms",
                detail: format!(
                    "atomic number is negative or too large: {}",
                    record.charge()
                ),
            })?;

        let coord = env.slice("PTR_COORD", record.coord_offset(), 3)?;
        let coord = [coord[0], coord[1], coord[2]];
        let (model, zeta, fractional_charge) = match record.nuclear_model_raw() {
            POINT_NUC => (NuclearModel::Point, None, None),
            GAUSSIAN_NUC => (
                NuclearModel::Gaussian,
                Some(env.slice("PTR_ZETA", record.zeta_offset(), 1)?[0]),
                None,
            ),
            FRAC_CHARGE_NUC => (
                NuclearModel::FiniteSpherical,
                None,
                Some(env.slice("PTR_FRAC_CHARGE", record.fractional_charge_offset(), 1)?[0]),
            ),
            other => {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("unsupported nuclear model {other}"),
                });
            }
        };

        let atom =
            Atom::try_new(atomic_number, coord, model, zeta, fractional_charge).map_err(|err| {
                cintxRsError::ChunkPlanFailed {
                    from: "raw_atoms",
                    detail: err.to_string(),
                }
            })?;
        atoms.push(atom);
    }

    let mut shells = Vec::new();
    shells
        .try_reserve_exact(bas.len())
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: bas.len().saturating_mul(size_of::<Shell>()),
        })?;

    for record in bas.iter() {
        let atom_index =
            u32::try_from(record.atom_index_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("negative shell atom index {}", record.atom_index_raw()),
            })?;
        let ang_momentum =
            u8::try_from(record.ang_momentum_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("invalid angular momentum {}", record.ang_momentum_raw()),
            })?;
        let nprim =
            u16::try_from(record.nprim_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: record.nprim_raw().unsigned_abs() as usize,
            })?;
        let nctr =
            u16::try_from(record.nctr_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: record.nctr_raw().unsigned_abs() as usize,
            })?;
        let kappa =
            i16::try_from(record.kappa_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("kappa does not fit i16: {}", record.kappa_raw()),
            })?;

        let exponents = Arc::<[f64]>::from(
            env.slice("PTR_EXP", record.exp_offset(), nprim as usize)?
                .to_vec()
                .into_boxed_slice(),
        );
        let coefficient_len = usize::from(nprim)
            .checked_mul(usize::from(nctr))
            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: "nprim*nctr overflowed usize".to_owned(),
            })?;
        // WR-03: the libcint env coefficient block is COLUMN-MAJOR — the value for
        // primitive `p` of contraction column `c` is stored at `env[c*nprim + p]`
        // (see CINTprim_to_ctr_0 in g1e.c: `c0 = coeff[nprim*i]`). cintx's internal
        // `Shell.coefficients` convention is ROW-MAJOR (`coeff[p*nctr + c]`, the
        // layout every launcher reads, e.g. `coefficients[ip*nctr_i + ci]`).
        // Transpose column-major → row-major here so the raw/ABI path agrees with
        // the safe-API path for general-contraction (nctr>1) shells. For nctr==1
        // (and nprim==1) the two layouts coincide, so this is byte-identical for the
        // single-contraction common case.
        let coeff_raw = env.slice("PTR_COEFF", record.coeff_offset(), coefficient_len)?;
        let nprim_usize = usize::from(nprim);
        let nctr_usize = usize::from(nctr);
        let mut coeff_rowmajor = vec![0.0_f64; coefficient_len];
        for c in 0..nctr_usize {
            for p in 0..nprim_usize {
                coeff_rowmajor[p * nctr_usize + c] = coeff_raw[c * nprim_usize + p];
            }
        }
        let coefficients = Arc::<[f64]>::from(coeff_rowmajor.into_boxed_slice());

        let shell = Shell::try_new(
            atom_index,
            ang_momentum,
            nprim,
            nctr,
            kappa,
            representation,
            exponents,
            coefficients,
        )
        .map_err(|err| cintxRsError::ChunkPlanFailed {
            from: "raw_shells",
            detail: err.to_string(),
        })?;
        shells.push(Arc::new(shell));
    }

    let basis = BasisSet::try_new(
        Arc::<[Atom]>::from(atoms.into_boxed_slice()),
        Arc::<[Arc<Shell>]>::from(shells.into_boxed_slice()),
    )
    .map_err(|err| cintxRsError::ChunkPlanFailed {
        from: "raw_basis",
        detail: err.to_string(),
    })?;

    let expected_arity = descriptor.entry.arity as usize;
    // The grids family follows libcint's `int1e_grids` calling convention, where
    // the shell tuple carries the bra/ket shells plus a trailing
    // `[grid_start, grid_end]` window: `[i, j, grid_start, grid_end]`. libcint
    // reads only `shls[0..2]` and loops the full grid set from `env` (NGRIDS);
    // the trailing window entries are ignored. Accept those two extra entries
    // and build the shell tuple from the leading `expected_arity` shells only,
    // matching libcint exactly (full env grid range, window entries ignored).
    let is_grids = descriptor.entry.canonical_family == "grids";
    let shell_count = if is_grids && shls.len() == expected_arity + 2 {
        expected_arity
    } else {
        shls.len()
    };
    if shell_count != expected_arity {
        return Err(cintxRsError::InvalidShellTuple {
            expected: expected_arity,
            got: shls.len(),
        });
    }

    let mut shell_indices = Vec::new();
    shell_indices.try_reserve_exact(shell_count).map_err(|_| {
        cintxRsError::HostAllocationFailed {
            bytes: shell_count.saturating_mul(size_of::<usize>()),
        }
    })?;
    for index in &shls[..shell_count] {
        let parsed = usize::try_from(*index).map_err(|_| cintxRsError::ChunkPlanFailed {
            from: "raw_shell_tuple",
            detail: format!("shell index must be non-negative: {index}"),
        })?;
        shell_indices.push(parsed);
    }

    let shell_tuple = basis
        .shell_tuple_for_indices(shell_indices)
        .map_err(|err| cintxRsError::ChunkPlanFailed {
            from: "raw_shell_tuple",
            detail: err.to_string(),
        })?;

    Ok((basis, shell_tuple))
}

fn map_resolver_error(api: RawApiId, err: ResolverError) -> cintxRsError {
    match err {
        ResolverError::MissingOperatorId(_) | ResolverError::MissingSymbol(_) => {
            cintxRsError::UnsupportedApi {
                requested: format!("raw api {} is missing from manifest", api.symbol()),
            }
        }
        ResolverError::MissingFamilyOperator { family, operator } => cintxRsError::UnsupportedApi {
            requested: format!("{family}/{operator}"),
        },
        ResolverError::UnsupportedRepresentation {
            family,
            operator,
            representation,
        } => cintxRsError::UnsupportedRepresentation {
            operator: format!("{family}/{operator}"),
            representation,
        },
    }
}

fn normalize_offset(
    slot: &'static str,
    offset: i32,
    env_len: usize,
) -> Result<usize, cintxRsError> {
    usize::try_from(offset).map_err(|_| cintxRsError::InvalidEnvOffset {
        slot,
        offset: env_len,
        env_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::required_f64s_for_bytes;

    struct RawFixture {
        shls_2: [i32; 2],
        shls_3: [i32; 3],
        atm: Vec<i32>,
        bas: Vec<i32>,
        env: Vec<f64>,
    }

    impl RawFixture {
        fn single_atom_three_shells() -> Self {
            // env layout:
            // 0..3 coordinates, 3 exp0, 4 coeff0, 5 exp1, 6 coeff1, 7 exp2, 8 coeff2
            let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6];
            let atm = vec![
                1, // charge / atomic number
                0, // PTR_COORD
                POINT_NUC, 0, // PTR_ZETA
                0, // PTR_FRAC_CHARGE
                0,
            ];
            let bas = vec![
                0, 0, 1, 1, 0, 3, 4, 0, // shell 0
                0, 1, 1, 1, 0, 5, 6, 0, // shell 1
                0, 0, 1, 1, 0, 7, 8, 0, // shell 2
            ];
            Self {
                shls_2: [0, 1],
                shls_3: [0, 1, 2],
                atm,
                bas,
                env,
            }
        }

        fn single_atom_four_shells() -> ([i32; 4], Vec<i32>, Vec<i32>, Vec<f64>) {
            let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
            let atm = vec![1, 0, POINT_NUC, 0, 0, 0];
            let bas = vec![
                0, 0, 1, 1, 0, 3, 4, 0, // shell 0
                0, 1, 1, 1, 0, 5, 6, 0, // shell 1
                0, 0, 1, 1, 0, 7, 8, 0, // shell 2
                0, 2, 1, 1, 0, 9, 10, 0, // shell 3
            ];
            ([0, 1, 2, 3], atm, bas, env)
        }
    }

    #[test]
    fn malformed_layouts_are_typed() {
        let err = RawAtmView::new(&[1, 2]).unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidAtmLayout { .. }));

        let err = RawBasView::new(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidBasLayout { .. }));
    }

    #[test]
    fn invalid_env_offsets_fail_validation() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut bas = fixture.bas.clone();
        bas[PTR_EXP] = 9999;
        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidEnvOffset { .. }));
    }

    #[test]
    fn invalid_dims_length_is_rejected_for_each_arity() {
        let fixture = RawFixture::single_atom_three_shells();

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&[1]),
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 1
            }
        ));

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                Some(&[1, 2]),
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 3,
                provided: 2
            }
        ));
    }

    #[test]
    fn undersized_output_buffer_is_reported() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut out = vec![0.0; 1];
        let err = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, cintxRsError::BufferTooSmall { .. }));
    }

    #[test]
    fn query_workspace_raw_and_eval_raw_none_match_workspace_expectations() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("query should succeed");

        let summary = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        }
        .expect("out == None should return requirements");

        assert_eq!(summary.not0, 0);
        assert_eq!(summary.bytes_written, 0);
        assert_eq!(summary.workspace_bytes, query.bytes);
    }

    #[test]
    fn memory_limit_hint_can_chunk_successfully() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(128));
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
            )
        }
        .expect("query should succeed with chunking");
        assert!(query.chunk_count >= 1);

        let mut out = vec![99.0; 3];
        let summary = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
                None,
            )
        }
        .expect("eval should succeed");

        assert!(summary.bytes_written > 0);
        // Kernel stubs write zeros to staging (real kernels come in Phase 9/10).
        // Verify eval_raw completed successfully and staging is populated (all zeros from stubs).
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn memory_limit_failure_keeps_output_slice_unchanged() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(1));
        let mut out = vec![7.0; 3];

        let err = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
                None,
            )
        }
        .unwrap_err();

        assert!(matches!(err, cintxRsError::MemoryLimitExceeded { .. }));
        assert!(
            out.iter().all(|value| *value == 7.0),
            "output slice unchanged on failure (no partial write)"
        );
    }

    #[test]
    fn cache_buffer_too_small_is_rejected_before_execution() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("query should succeed");

        let required_cache = required_f64s_for_bytes(query.bytes).expect("cache conversion");
        let mut out = vec![0.0; 3];
        let mut cache = vec![0.0; required_cache.saturating_sub(1)];
        let err = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                Some(&mut cache),
            )
        }
        .unwrap_err();

        assert!(matches!(err, cintxRsError::BufferTooSmall { .. }));
    }

    #[test]
    fn three_center_contract_query_and_eval_work_for_supported_backend() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                None,
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("3c query should still resolve and plan");
        assert_eq!(query.work_units, 3);

        let mut out = vec![1.0; 3];
        let summary = unsafe {
            eval_raw(
                RawApiId::INT3C1E_P2_CART,
                Some(&mut out),
                None,
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        }
        .expect("3c eval should succeed when kernel support is available");
        assert!(summary.bytes_written > 0);
        // Kernel stubs write zeros to staging (real kernels come in Phase 9/10).
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn f12_cart_symbol_is_rejected_with_explicit_sph_envelope_reason() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_cart"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested } if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(not(feature = "with-f12"))]
    #[test]
    fn f12_sph_symbol_requires_with_f12_profile() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("active profile")
                    && requested.contains(active_manifest_profile())
        ));
    }

    #[cfg(feature = "with-f12")]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_reports_with_f12_sph_envelope_for_cart_representation() {
        let descriptor = Resolver::descriptor_by_symbol("int2e_stg_sph")
            .expect("stg symbol must exist in manifest");
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Cart,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Cart,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested } if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(feature = "with-f12")]
    #[test]
    fn f12_sph_symbol_is_queryable_when_feature_enabled() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .expect("with-f12 should allow sph-only f12 symbols");
        assert!(query.bytes > 0);
    }

    #[cfg(not(feature = "with-4c1e"))]
    #[test]
    fn int4c1e_requires_with_4c1e_profile() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("active profile")
                    && requested.contains(active_manifest_profile())
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn int4c1e_rejects_bug_envelope_inputs() {
        let (shls_4, atm, mut bas, env) = RawFixture::single_atom_four_shells();
        bas[ANG_OF] = 5; // max(l)>4 should fail the Validated4C1E envelope.

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_reports_validated_4c1e_reason_for_out_of_envelope_shells() {
        let descriptor = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("int4c1e cart symbol must exist in manifest");
        let (shls_4, atm, mut bas, env) = RawFixture::single_atom_four_shells();
        bas[ANG_OF] = 5;

        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Cart,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Cart,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn int4c1e_accepts_validated_inputs() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .expect("validated 4c1e envelope should be queryable");
        assert!(query.bytes > 0);
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_rejects_source_only_symbol_without_unstable_feature() {
        // int2e_ipip1_sph was promoted to stable in Phase 25 HESS-02 (D-07); use a
        // still-source-only symbol to exercise the unstable-feature gate.
        let descriptor = Resolver::descriptor_by_symbol("int2e_breit_r1p2_spinor")
            .expect("source-only symbol must exist in manifest");
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Spheric,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Spheric,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("requires feature `unstable-source-api`")
        ));
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test]
    fn source_only_symbol_requires_unstable_feature() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                // int2e_ipip1_sph promoted to stable (Phase 25 HESS-02 D-07); use a
                // still-source-only symbol to exercise the unstable-feature gate.
                RawApiId::Symbol("int2e_breit_r1p2_spinor"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("requires feature `unstable-source-api`")
        ));
    }

    /// Verify that eval_raw() uses direct executor.execute() with an owned staging buffer,
    /// not RecordingExecutor. This is a compile-time and runtime guarantee: RecordingExecutor
    /// no longer exists in this module, and the staging path is exercised directly.
    #[test]
    fn eval_raw_reads_staging_directly() {
        let fixture = RawFixture::single_atom_three_shells();
        // Allocate enough output for a 2-shell 1e integral (int1e_ovlp_cart: 2-center, cart).
        // Shell 0 has ang=0 (1 AO), shell 1 has ang=1 (3 AOs). Output size = 1 * 3 = 3 elements.
        let mut out = vec![0.0f64; 3];
        let result = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        };
        // eval_raw must succeed: the direct staging path is exercised end-to-end.
        // bytes_written > 0 confirms the staging buffer was written and output was committed.
        let summary = result.expect("eval_raw_reads_staging_directly should succeed");
        assert!(
            summary.bytes_written > 0,
            "bytes_written must be > 0 (staging path was exercised): bytes_written={}",
            summary.bytes_written
        );
    }

    /// Verify that eval_raw returns InvalidEnvParam when env[PTR_F12_ZETA] is 0.0
    /// for an F12 symbol. This tests that the raw compat path calls validate_f12_env_params.
    #[cfg(feature = "with-f12")]
    #[test]
    fn eval_raw_f12_symbol_with_zero_zeta_returns_invalid_env_param() {
        let (shls_4, atm, bas, _env) = RawFixture::single_atom_four_shells();
        // Construct env with env[PTR_F12_ZETA=9] = 0.0 (invalid zeta).
        // The error fires before execution, so we only need valid enough env for
        // descriptor lookup and plan construction. Use the four-shell env layout
        // with zeta forced to 0 at index 9.
        let mut env_full = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.4];
        env_full[PTR_F12_ZETA] = 0.0;
        let mut out = vec![0.0f64; 16]; // 2x2x2x2 output upper bound
        let err = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                Some(&mut out),
                None,
                &shls_4,
                &atm,
                &bas,
                &env_full,
                None,
                None,
            )
        }
        .unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_F12_ZETA"),
            "expected InvalidEnvParam(PTR_F12_ZETA) for zero zeta, got: {err:?}"
        );
    }

    /// Verify that eval_raw passes env param validation (no InvalidEnvParam) when
    /// env[PTR_F12_ZETA] is non-zero for an F12 symbol. The call may fail later at
    /// UnsupportedApi or executor level (no GPU in test), but the zeta gate must pass.
    #[cfg(feature = "with-f12")]
    #[test]
    fn eval_raw_f12_symbol_with_valid_zeta_passes_env_param_validation() {
        let (shls_4, atm, bas, mut env_full) = RawFixture::single_atom_four_shells();
        env_full[PTR_F12_ZETA] = 1.2; // valid non-zero zeta
        let mut out = vec![0.0f64; 16];
        let result = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                Some(&mut out),
                None,
                &shls_4,
                &atm,
                &bas,
                &env_full,
                None,
                None,
            )
        };
        // Must not be InvalidEnvParam — that would mean our validation is wrong.
        assert!(
            !matches!(result, Err(cintxRsError::InvalidEnvParam { .. })),
            "eval_raw should not return InvalidEnvParam when zeta=1.2: {result:?}"
        );
    }

    #[cfg(feature = "unstable-source-api")]
    #[test]
    fn eval_raw_grids_symbol_with_missing_grids_params_returns_invalid_env_param() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut out = vec![0.0_f64; 256];
        let err = unsafe {
            eval_raw(
                RawApiId::Symbol("int1e_grids_sph"),
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        }
        .unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "NGRIDS"),
            "expected InvalidEnvParam(NGRIDS), got: {err:?}"
        );
    }

    #[cfg(feature = "unstable-source-api")]
    #[test]
    fn eval_raw_grids_symbol_with_valid_grids_params_passes_env_validation() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut env_full = fixture.env.clone();
        if env_full.len() <= PTR_GRIDS {
            env_full.resize(PTR_GRIDS + 1, 0.0);
        }
        let ptr_grids = env_full.len();
        env_full[NGRIDS] = 1.0;
        env_full[PTR_GRIDS] = ptr_grids as f64;
        env_full.extend_from_slice(&[0.0, 0.0, 0.0]);

        let mut out = vec![0.0_f64; 256];
        let result = unsafe {
            eval_raw(
                RawApiId::Symbol("int1e_grids_sph"),
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &env_full,
                None,
                None,
            )
        };
        assert!(
            !matches!(result, Err(cintxRsError::InvalidEnvParam { .. })),
            "eval_raw grids path should not fail env validation when grids params are set: {result:?}"
        );
    }

    // --- Phase 19 D-05: ECP slot constants + EcpBasArray + dispatch arm ---

    #[test]
    fn ecp_slot_constants_match_pyscf_nr_ecp_h() {
        assert_eq!(RADI_POWER, 3);
        assert_eq!(SO_TYPE_OF, 4);
        assert_eq!(AS_ECPBAS_OFFSET, 18);
        assert_eq!(AS_NECPBAS, 19);
        assert_eq!(ECP_LMAX, 5);
    }

    #[test]
    fn ecp_bas_array_accepts_slab_with_bas_slots_multiple_length() {
        // 2 rows × 8 slots = 16 i32s — should succeed.
        let slab = [0i32; 16];
        let view = EcpBasArray::new(&slab).expect("16-slot slab should be accepted");
        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
    }

    #[test]
    fn ecp_bas_array_rejects_non_multiple_length() {
        // 9 i32s is not a multiple of BAS_SLOTS=8 — should be rejected
        // with InvalidBasLayout (same variant RawBasView::new uses).
        let slab = [0i32; 9];
        let err = EcpBasArray::new(&slab).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: 9,
            }
        ));
    }

    #[test]
    fn ecp_bas_array_named_getters_read_correct_slots() {
        // Build one ecpbas row with radial_power=2 at slot 3 and so_type=0
        // at slot 4. Layout (BAS_SLOTS=8): [atom, ang, nprim, RADI_POWER,
        // SO_TYPE_OF, ptr_exp, ptr_coeff, _].
        let row: [i32; 8] = [
            0, /* atom */
            -1, /* ang (Local sentinel) */
            1, /* nprim */
            2, /* RADI_POWER */
            0, /* SO_TYPE_OF (scalar) */
            10, /* PTR_EXP */
            11, /* PTR_COEFF */
            0, /* padding */
        ];
        let view = EcpBasArray::new(&row).expect("single-row slab");
        assert_eq!(view.len(), 1);
        assert_eq!(view.radial_power(0), 2);
        assert_eq!(view.so_type(0), 0);
        // iter_rows yields the same 8-slot record.
        let mut iter = view.iter_rows();
        let first = iter.next().expect("at least one row");
        assert_eq!(first.len(), BAS_SLOTS);
        assert_eq!(first[RADI_POWER], 2);
        assert_eq!(first[SO_TYPE_OF], 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn raw_api_id_ecp_constants_expose_canonical_symbols() {
        // Construct via the public constants and round-trip through symbol().
        assert_eq!(
            RawApiId::INT1E_ECP_CART.symbol(),
            "int1e_ecp_cart",
        );
        assert_eq!(RawApiId::INT1E_ECP_SPH.symbol(), "int1e_ecp_sph");
        assert_eq!(
            RawApiId::INT1E_ECP_IPNUC_CART.symbol(),
            "int1e_ecp_ipnuc_cart",
        );
        assert_eq!(
            RawApiId::INT1E_ECP_IPNUC_SPH.symbol(),
            "int1e_ecp_ipnuc_sph",
        );
    }

    #[test]
    fn is_ecp_family_symbol_matches_only_int1e_ecp_prefix() {
        assert!(is_ecp_family_symbol("int1e_ecp_cart"));
        assert!(is_ecp_family_symbol("int1e_ecp_sph"));
        assert!(is_ecp_family_symbol("int1e_ecp_ipnuc_cart"));
        assert!(is_ecp_family_symbol("int1e_ecp_ipnuc_sph"));
        // Negative cases — non-ECP symbols must not match.
        assert!(!is_ecp_family_symbol("int1e_ovlp_cart"));
        assert!(!is_ecp_family_symbol("int1e_nuc_sph"));
        assert!(!is_ecp_family_symbol("int4c1e_cart"));
        assert!(!is_ecp_family_symbol("int2e_stg_sph"));
    }

    #[test]
    fn eval_raw_ecp_symbol_with_zero_necpbas_returns_invalid_env_param() {
        // Build an env that is long enough to reach AS_NECPBAS=19 but
        // explicitly sets env[AS_NECPBAS] = 0.0 (no ecpbas rows attached).
        // The ECP dispatch guard must fire before kernel launch, returning
        // cintxRsError::InvalidEnvParam { param: "AS_NECPBAS", ... }.
        let fixture = RawFixture::single_atom_three_shells();
        let mut env_full = fixture.env.clone();
        // Pad up to (AS_NECPBAS + 1) so env.get(AS_NECPBAS) is in bounds.
        if env_full.len() <= AS_NECPBAS {
            env_full.resize(AS_NECPBAS + 1, 0.0);
        }
        env_full[AS_NECPBAS] = 0.0; // explicit zero — guard should fire
        let mut out = vec![0.0_f64; 64];
        let err = unsafe {
            eval_raw(
                RawApiId::INT1E_ECP_SPH,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &env_full,
                None,
                None,
            )
        }
        .unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "AS_NECPBAS"),
            "expected InvalidEnvParam(AS_NECPBAS) for zero necpbas, got: {err:?}"
        );
    }

    // --- PTR_RINV_ORIG (Plan 21-01) tests ---

    /// Verify that PTR_RINV_ORIG is the correct libcint constant value (4).
    #[test]
    fn ptr_rinv_orig_is_4() {
        assert_eq!(PTR_RINV_ORIG, 4, "PTR_RINV_ORIG must equal 4 (libcint constant)");
    }

    /// Verify that is_iprinv_family_symbol detects iprinv symbols correctly.
    #[test]
    fn is_iprinv_family_symbol_detects_iprinv() {
        assert!(is_iprinv_family_symbol("int1e_iprinv_sph"));
        assert!(is_iprinv_family_symbol("int1e_iprinv_cart"));
        assert!(is_iprinv_family_symbol("int1e_ecp_iprinv_sph"));
        assert!(is_iprinv_family_symbol("ECPscalar_iprinv_sph"));
        // Sanity: non-iprinv symbols must not match
        assert!(!is_iprinv_family_symbol("int1e_ovlp_sph"));
        assert!(!is_iprinv_family_symbol("int1e_ipnuc_sph"));
        assert!(!is_iprinv_family_symbol("int2e_sph"));
    }

    /// Verify that is_iprinv_family_symbol does NOT match non-iprinv ip* symbols.
    #[test]
    fn is_iprinv_family_symbol_does_not_match_ipovlp_ipkin_ipnuc() {
        assert!(!is_iprinv_family_symbol("int1e_ipovlp_sph"));
        assert!(!is_iprinv_family_symbol("int1e_ipkin_sph"));
        assert!(!is_iprinv_family_symbol("int1e_ipnuc_sph"));
        assert!(!is_iprinv_family_symbol("int2e_ip1_sph"));
    }
}
