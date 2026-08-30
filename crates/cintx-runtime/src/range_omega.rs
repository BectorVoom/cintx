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

/// The `ng[]` angular-momentum raises a derivative row applies, **per shell of
/// the tuple** rather than per `CINTg0_2e` slot.
///
/// `CINTinit_int2e_EnvVars` builds the G tensor at
/// `l_ceil = l + ng[{I,J,K,L}INC]` and takes `rys_order` from the raised sum
/// (`g2e.c:74-79`), so a derivative row's `rys_order` is NOT `(Σ l)/2 + 1`.
/// That is precisely why [`supports_range_omega`] was narrow before D-PBC-24
/// P2-1: an unraised estimate would size the workspace for fewer Rys roots than
/// the kernel then writes.
///
/// Indexing is by TUPLE POSITION, which makes one table serve all three
/// families even though they map onto the four `CINTg0_2e` slots differently
/// (`int3c2e` puts its auxiliary shell in the `ll` slot with `lk_ceil = 0`;
/// `int2c2e` zeroes `lj_ceil`/`ll_ceil`). Only the SUM enters `rys_order`, so
/// the slot permutation does not matter here — but it does in the launchers,
/// and each of these entries is the mirror of a `build_2e_shape(...)` call
/// there:
///
/// | family | tuple | site |
/// |---|---|---|
/// | `2e` | `(i, j, k, l)` | `Hess2eKind::headroom`, `launch_two_electron_ip{1,2}` |
/// | `3c2e` | `(i, j, aux)` | `launch_center_3c2e_{ip1,ip2,hess}` |
/// | `2c2e` | `(i, k)` | `launch_center_2c2e_{grad,hess}` |
///
/// Returns `None` for an operator this table does not cover, which
/// [`supports_range_omega`] treats as "not admitted" — a refusal, never a
/// guess at the raises.
pub fn derivative_headroom(
    canonical_family: &str,
    operator_name: &str,
) -> Option<&'static [usize]> {
    match canonical_family {
        // (i, j, k, l)
        "2e" => Some(match operator_name {
            "electron-repulsion" => &[0, 0, 0, 0],
            "ip1" => &[1, 0, 0, 0],
            "ip2" => &[0, 0, 1, 0],
            "ipip1" => &[2, 0, 0, 0],
            "ipvip1" => &[1, 1, 0, 0],
            "ip1ip2" => &[1, 0, 1, 0],
            "ipip1ipip2" => &[2, 0, 2, 0],
            "ipvip1ipvip2" => &[1, 1, 1, 1],
            _ => return None,
        }),
        // (i, j, aux) — the auxiliary shell occupies the 2e `ll` slot.
        "3c2e" => Some(match operator_name {
            "electron-repulsion" => &[0, 0, 0],
            "ip1" => &[1, 0, 0],
            "ip2" => &[0, 0, 1],
            "ipip1" => &[2, 0, 0],
            "ipip2" => &[0, 0, 2],
            "ipvip1" => &[1, 1, 0],
            "ip1ip2" => &[1, 0, 1],
            _ => return None,
        }),
        // (i, k) — both kets are phantom s.
        "2c2e" => Some(match operator_name {
            "electron-repulsion" => &[0, 0],
            "ip1" => &[1, 0],
            "ip2" => &[0, 1],
            "ipip1" => &[2, 0],
            "ip1ip2" => &[1, 1],
            _ => return None,
        }),
        _ => None,
    }
}

/// Whether this `(canonical_family, operator_name)` pair honours `range_omega`.
///
/// # Scope
///
/// Everything that shares `CINTg0_2e` (`g2e.c:171`, `g3c2e.c:131`,
/// `g2c2e.c:104`) reads `env[8]` upstream, so in principle every row of the
/// `2e`, `3c2e` and `2c2e` families could be served. What gates the scope here
/// is not the kernel but the WORKSPACE: `rys_order` has to be known exactly
/// before evaluation, because short range doubles the Rys roots and the root
/// count sizes the G tensor. So a row is admitted exactly when
/// [`derivative_headroom`] knows its `ng[]` raises.
///
/// Admitted, as of D-PBC-24 P2-1:
///
/// * the scalar Coulomb rows `int2e`, `int3c2e`, `int2c2e` (stage 2) — the set
///   `aux_e2` (`int3c2e` + `int2c2e`) needs for RSDF/RSMDF, and `int2e` for
///   `rsjk` and molecular RSH;
/// * their `ip`-family gradient and Hessian rows, which is what a
///   range-separated GRADIENT needs.
///
/// Still refused, and deliberately:
///
/// * the GIAO/gauge rows of the `2e` family (`g1`, `gg1`, `g1g2`, `ig1`,
///   `ipvg{1,2}_xp1`, `ip1v_r{c,}1`). They read `env[8]` upstream and their
///   launchers are host-routed, so they would very likely just work — but each
///   carries its own `common_factor` scale and position-operator composition,
///   and none is gated against the vendor under a set ω. Widening a scope
///   without extending the gate is how a full-range substitute ships.
/// * the relativistic σ·p / σ·r spinor rows (`spsp1`, `ipspsp1`, `srsr1`, …),
///   for the same reason.
/// * every other family: `f12` (whose `CINTg0_2e_stg`/`_yp` have no omega
///   branch at all), `breit`, `origi`, `origk`, `ssc`, `4c1e`, `1e`, `3c1e`,
///   `ecp`, `grids`.
///
/// A refused row returns `UnsupportedApi` rather than silently evaluating the
/// full-range operator. A full-range substitute runs, converges, and is
/// silently a different method; that is the one outcome D-PBC-24 forbids in
/// writing.
pub fn supports_range_omega(canonical_family: &str, operator_name: &str) -> bool {
    derivative_headroom(canonical_family, operator_name).is_some()
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
/// Exact for the SCALAR rows, where every `l_ceil` equals the shell's own
/// angular momentum. `int3c2e` folds its auxiliary shell into the `ll` slot
/// with `lk_ceil = 0` and `int2c2e` zeroes `lj_ceil`/`ll_ceil`, so in all three
/// cases the sum is simply the sum over the shells actually present in the
/// tuple (`g2e.c:74-77`, `g3c2e.c:70`, `g2c2e.c:60`).
///
/// Derivative rows raise it; use [`rys_order_with_headroom`].
pub fn rys_order_for_angular_momenta(angular_momenta: impl IntoIterator<Item = usize>) -> usize {
    angular_momenta.into_iter().sum::<usize>() / 2 + 1
}

/// [`rys_order_for_angular_momenta`] with a derivative row's `ng[]` raises
/// applied, `headroom` indexed by tuple position as
/// [`derivative_headroom`] returns it.
///
/// A shorter `headroom` than `angular_momenta` raises nothing on the tail,
/// which is what makes the scalar `&[0, 0, ..]` entries and a mismatched arity
/// both behave conservatively rather than panicking.
pub fn rys_order_with_headroom(
    angular_momenta: impl IntoIterator<Item = usize>,
    headroom: &[usize],
) -> usize {
    angular_momenta
        .into_iter()
        .enumerate()
        .map(|(idx, l)| l + headroom.get(idx).copied().unwrap_or(0))
        .sum::<usize>()
        / 2
        + 1
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

    /// The scope is the scalar Coulomb rows plus their `ip`-family derivatives,
    /// and nothing else.
    ///
    /// The exclusions matter as much as the inclusions: `g1` and `spsp1` share
    /// `CINTg0_2e` with `ip1` and would very likely just work, and they are
    /// refused anyway because nothing gates them under a set ω (see
    /// [`supports_range_omega`]'s docs). This test is what stops the scope from
    /// drifting past the gate.
    #[test]
    fn the_scope_is_the_scalar_coulomb_rows_plus_their_ip_derivatives() {
        for family in ["2e", "3c2e", "2c2e"] {
            assert!(supports_range_omega(family, "electron-repulsion"), "{family}");
            for op in ["ip1", "ip2", "ipip1", "ip1ip2"] {
                assert!(supports_range_omega(family, op), "{family}:{op}");
            }
        }
        assert!(supports_range_omega("2e", "ipvip1"));
        assert!(supports_range_omega("2e", "ipip1ipip2"));
        assert!(supports_range_omega("2e", "ipvip1ipvip2"));
        assert!(supports_range_omega("3c2e", "ipvip1"));
        assert!(supports_range_omega("3c2e", "ipip2"));

        // 2c2e has only centres i and k — no `ipvip1`, and no `ipip2`.
        assert!(!supports_range_omega("2c2e", "ipvip1"));
        assert!(!supports_range_omega("2c2e", "ipip2"));
        // GIAO/gauge and relativistic spinor rows: not gated, so not admitted.
        for op in ["g1", "gg1", "g1g2", "ig1", "ip1v_r1", "spsp1", "ipspsp1"] {
            assert!(!supports_range_omega("2e", op), "2e:{op}");
        }
        // Other families entirely.
        assert!(!supports_range_omega("1e", "electron-repulsion"));
        assert!(!supports_range_omega("f12", "stg"));
        assert!(!supports_range_omega("ecp", "ecp"));
        assert!(!supports_range_omega("grids", "grids"));
    }

    /// Each headroom entry has one raise per shell of its family's tuple, and
    /// the scalar rows raise nothing.
    ///
    /// The arity check is the one that matters: an entry one element short
    /// would silently under-raise the LAST shell — the auxiliary one for
    /// `3c2e`, which is exactly the position `ip2` differentiates.
    #[test]
    fn the_headroom_table_has_one_raise_per_tuple_position() {
        for (family, arity) in [("2e", 4usize), ("3c2e", 3), ("2c2e", 2)] {
            let scalar = derivative_headroom(family, "electron-repulsion").unwrap();
            assert_eq!(scalar.len(), arity, "{family} scalar arity");
            assert!(scalar.iter().all(|&r| r == 0), "{family} scalar raises nothing");

            for op in ["ip1", "ip2", "ipip1", "ip1ip2"] {
                let h = derivative_headroom(family, op).unwrap();
                assert_eq!(h.len(), arity, "{family}:{op} arity");
                assert!(h.iter().sum::<usize>() > 0, "{family}:{op} must raise something");
            }
        }
        assert_eq!(derivative_headroom("2e", "unknown-row"), None);
        assert_eq!(derivative_headroom("f12", "ip1"), None);
    }

    /// `rys_order` with the raises applied — the number the workspace is sized
    /// from, and the whole reason the derivative rows were out of scope before.
    #[test]
    fn headroom_raises_the_rys_order() {
        // int2e_ip1 over four p shells: (1+1 + 1 + 1 + 1)/2 + 1 = 3, where the
        // scalar row is (1+1+1+1)/2 + 1 = 3 as well — same here, so pick a case
        // where they differ.
        assert_eq!(rys_order_for_angular_momenta([1, 1, 1, 1]), 3);
        assert_eq!(rys_order_with_headroom([1, 1, 1, 1], &[2, 0, 0, 0]), 4);
        // Short range then doubles the RAISED order, not the bare one.
        assert_eq!(nrys_roots_for(3, Some(-0.8)), 6);
        // int3c2e_ip2 on (s,s|d): (0 + 0 + 2+1)/2 + 1 = 2.
        assert_eq!(rys_order_with_headroom([0, 0, 2], &[0, 0, 1]), 2);
        // A short headroom raises nothing on the tail rather than panicking.
        assert_eq!(rys_order_with_headroom([1, 1, 1, 1], &[1]), 3);
    }
}
