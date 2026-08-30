//! Range-separated Coulomb support: the `env[PTR_RANGE_OMEGA]` (`env[8]`) contract.
//!
//! # D-PBC-24
//!
//! libcint has **no** `int2e_sr_*` symbol. Range separation is a *parameter* on
//! the ordinary Coulomb operator, read from `env[PTR_RANGE_OMEGA]` inside
//! `CINTg0_2e` (`libcint-master/src/g2e.c:4443-4512`), which is shared verbatim
//! by `int2e` (`g2e.c:171`), `int3c2e` (`g3c2e.c:131`) and `int2c2e`
//! (`g2c2e.c:104`). PySCF drives it exactly that way — `pyscf/pbc/df/rsjk.py:186`
//! sets `supmol_sr.omega = -self.omega` and then calls the *standard* `int2e`.
//! cintx therefore carries omega as an [`crate::options::ExecutionOptions`] /
//! [`crate::planner::OperatorEnvParams`] field, never as a new operator symbol.
//!
//! ## Sign convention (libcint's, and `pyscf_pbc_df::traits::JkOpts::omega`'s)
//!
//! | `range_omega` | operator |
//! |---|---|
//! | `None` or `Some(0.0)` | full Coulomb `1/r₁₂` |
//! | `Some(ω)` with `ω > 0` | long range, `erf(ω r₁₂)/r₁₂` |
//! | `Some(ω)` with `ω < 0` | short range, `erfc(|ω| r₁₂)/r₁₂` |
//!
//! ## Why this lives in the control plane and not only in the kernels
//!
//! Short range **doubles the Rys root count** for `rys_order <= 3`
//! (`g2e.c:76-79`, `g3c2e.c:70-77`, `g2c2e.c:61-68`), and `nrys_roots` is what
//! sets `g_stride_i`, `g_stride_k` and `g_size` — the workspace. So
//! `query_workspace` has to see omega, or the query/evaluate capability token
//! (D-08) is wrong the moment a caller asks for short range. [`nrys_roots_for`]
//! is the single place that rule is expressed; the planner and every kernel
//! prologue call it.

/// libcint `PTR_RANGE_OMEGA` — index of the range-separation parameter in `env`.
///
/// Source: `libcint-master/include/cint.h.in`. Mirrored (as a `usize`) by
/// `cintx_compat::raw::PTR_RANGE_OMEGA`.
pub const PTR_RANGE_OMEGA: usize = 8;

/// libcint `EXPCUTOFF_SR` (`libcint-master/src/rys_roots.h:46`).
///
/// The short-range Rys kernel is numerically unstable once `theta * x` grows
/// past this, so `CINTg0_2e` returns "no contribution" instead of evaluating
/// it. This is **part of the algorithm**, not an optimisation — porting it is
/// required for parity, not just for speed.
pub const EXPCUTOFF_SR: f64 = 40.0;

/// The largest `rys_order` whose short-range integral libcint computes as
/// "full minus long range" with doubled roots (`g2e.c:78`).
///
/// Above it, `CINTsr_rys_roots` — the lower-bounded `∫_lower^1` quadrature
/// family — is required instead. cintx carries that family since D-PBC-24
/// stage 3 (`cintx_cubecl::math::rys_wheeler::sr_rys_roots_host`), so both
/// regimes evaluate; this constant is only the boundary between them, not a
/// support boundary.
pub const SR_DOUBLED_ROOT_MAX_ORDER: usize = 3;

/// `nrys_roots` for a given `rys_order` under a range-separation parameter.
///
/// Verbatim port of the rule repeated in `g2e.c:76-79`, `g3c2e.c:70-77` and
/// `g2c2e.c:61-68`:
///
/// ```text
/// int nrys_roots = rys_order;
/// double omega = env[PTR_RANGE_OMEGA];
/// if (omega < 0 && rys_order <= 3) {
///         nrys_roots *= 2;
/// }
/// ```
///
/// `omega >= 0` (long range and full range) never changes the root count; only
/// short range does, and only in the doubled-root regime.
pub fn nrys_roots_for(rys_order: usize, range_omega: Option<f64>) -> usize {
    match range_omega {
        Some(omega) if omega < 0.0 && rys_order <= SR_DOUBLED_ROOT_MAX_ORDER => rys_order * 2,
        _ => rys_order,
    }
}

/// True when `range_omega` selects the short-range (`erfc`) operator.
pub fn is_short_range(range_omega: Option<f64>) -> bool {
    matches!(range_omega, Some(omega) if omega < 0.0)
}

/// True when `range_omega` selects a non-Coulomb (range-separated) operator at
/// all, i.e. the kernel prologue must take a branch other than `omega == 0`.
pub fn is_range_separated(range_omega: Option<f64>) -> bool {
    matches!(range_omega, Some(omega) if omega != 0.0)
}

/// Whether this `(canonical_family, operator_name)` pair honours `range_omega`.
///
/// # Scope (D-PBC-24 stage 2)
///
/// The three scalar Coulomb operators — `int2e`, `int3c2e`, `int2c2e` — are the
/// ones that share `CINTg0_2e` *and* whose `rys_order` is exactly
/// `(Σ l)/2 + 1`, which is what makes the workspace estimate below exact. They
/// are also the complete set the range-separated consumers need: `aux_e2`
/// (`int3c2e` + `int2c2e`) for RSDF/RSMDF, and `int2e` for `rsjk` and for
/// molecular RSH.
///
/// Every other operator — the `ip1`/`ipip1`/… derivative rows of the same
/// families, and the `f12`, `breit`, `origi`, `origk`, `ssc`, `4c1e`, `1e`,
/// `3c1e`, `ecp` and `grids` families — **rejects** a set `range_omega` rather
/// than silently evaluating the full-range operator. A full-range substitute
/// runs, converges, and is silently a different method; that is the one outcome
/// D-PBC-24 forbids in writing.
pub fn supports_range_omega(canonical_family: &str, operator_name: &str) -> bool {
    matches!(canonical_family, "2e" | "3c2e" | "2c2e") && operator_name == "electron-repulsion"
}

/// Whether libcint's counterpart of this family READS `env[PTR_RANGE_OMEGA]`.
///
/// This is a statement about the *reference*, not about cintx, and it is what
/// the raw compat path needs: a caller working inside a PySCF-style
/// `range_coulomb(omega)` block leaves `env[8]` set for every integral it
/// evaluates in that block, including ones libcint itself ignores it for.
/// Silently ignoring it where libcint ignores it is correct; silently ignoring
/// it where libcint honours it is the "different method" failure D-PBC-24
/// exists to prevent, so those families are routed into
/// [`supports_range_omega`] and refused when unimplemented.
///
/// The readers, verified by grep over `libcint-master/src`:
///
/// * `CINTg0_2e` (`g2e.c:4443`) — shared by `CINTinit_int2e_EnvVars`
///   (`g2e.c:171`, i.e. the whole `int2e_*` symbol space: the `2e`, `4c1e`,
///   `breit`, `origi`, `origk` and `ssc` canonical families),
///   `CINTinit_int3c2e_EnvVars` (`g3c2e.c:131`) and
///   `CINTinit_int2c2e_EnvVars` (`g2c2e.c:104`)
/// * `g1e_grids.c:31,98` — the `int1e_grids` family
///
/// Non-readers: the `1e`, `3c1e` and `ecp` families, and `f12` — whose
/// `CINTg0_2e_stg` / `CINTg0_2e_yp` (`g2e_f12.c:95,151`) have no omega branch
/// at all.
pub fn family_consumes_range_omega(canonical_family: &str) -> bool {
    matches!(
        canonical_family,
        "2e" | "3c2e" | "2c2e" | "4c1e" | "breit" | "origi" | "origk" | "ssc" | "grids"
    )
}

/// `rys_order = (Σ l_ceil)/2 + 1` for a scalar Coulomb shell tuple.
///
/// Exact for the operators [`supports_range_omega`] admits, where every
/// `l_ceil` equals the shell's own angular momentum (no `ng[IINC]` raises).
/// `int3c2e` folds its auxiliary shell into the `ll` slot with `lk_ceil = 0`
/// and `int2c2e` zeroes `lj_ceil`/`ll_ceil`, so in all three cases the sum is
/// simply the sum over the shells actually present in the tuple
/// (`g2e.c:74-77`, `g3c2e.c:70`, `g2c2e.c:60`).
pub fn rys_order_for_angular_momenta(angular_momenta: impl IntoIterator<Item = usize>) -> usize {
    angular_momenta.into_iter().sum::<usize>() / 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_long_range_never_double_the_roots() {
        for rys_order in 1..=8 {
            assert_eq!(nrys_roots_for(rys_order, None), rys_order);
            assert_eq!(nrys_roots_for(rys_order, Some(0.0)), rys_order);
            assert_eq!(nrys_roots_for(rys_order, Some(0.8)), rys_order);
        }
    }

    #[test]
    fn short_range_doubles_only_through_order_three() {
        // g2e.c:78 — `omega < 0 && rys_order <= 3`.
        assert_eq!(nrys_roots_for(1, Some(-0.8)), 2);
        assert_eq!(nrys_roots_for(2, Some(-0.8)), 4);
        assert_eq!(nrys_roots_for(3, Some(-0.8)), 6);
        assert_eq!(nrys_roots_for(4, Some(-0.8)), 4);
        assert_eq!(nrys_roots_for(5, Some(-0.8)), 5);
    }

    #[test]
    fn rys_order_matches_the_reference_cells() {
        // D-PBC-24 §3.4, the table Phase 14 gates on.
        // int3c2e, He-fcc sto-3g + aux: l_i = l_j = 0, l_k = 2.
        assert_eq!(rys_order_for_angular_momenta([0, 0, 2]), 2);
        // int3c2e, diamond gth-szv + aux: l_i = l_j = 1, l_k = 2.
        assert_eq!(rys_order_for_angular_momenta([1, 1, 2]), 3);
        // int2c2e, either auxcell: l_i = l_k = 2.
        assert_eq!(rys_order_for_angular_momenta([2, 2]), 3);
        // int2e (rsjk), s/p basis.
        assert_eq!(rys_order_for_angular_momenta([1, 1, 1, 1]), 3);
        // int2e (rsjk), d functions — the sr_rys_roots regime.
        assert_eq!(rys_order_for_angular_momenta([2, 2, 2, 2]), 5);
    }

    #[test]
    fn libcint_omega_readers_are_the_2e_derived_families_plus_grids() {
        for family in [
            "2e", "3c2e", "2c2e", "4c1e", "breit", "origi", "origk", "ssc", "grids",
        ] {
            assert!(family_consumes_range_omega(family), "{family}");
        }
        for family in ["1e", "3c1e", "ecp", "f12", "helper"] {
            assert!(!family_consumes_range_omega(family), "{family}");
        }
    }

    #[test]
    fn only_the_three_scalar_coulomb_operators_take_omega() {
        assert!(supports_range_omega("2e", "electron-repulsion"));
        assert!(supports_range_omega("3c2e", "electron-repulsion"));
        assert!(supports_range_omega("2c2e", "electron-repulsion"));
        assert!(!supports_range_omega("2e", "ip1"));
        assert!(!supports_range_omega("3c2e", "ip2"));
        assert!(!supports_range_omega("1e", "electron-repulsion"));
        assert!(!supports_range_omega("f12", "stg"));
        assert!(!supports_range_omega("ecp", "ecp"));
        assert!(!supports_range_omega("grids", "grids"));
    }
}
